"""
Parallel self-play environment for training (workers = separate processes, each with several arenas).

Per arena the LEARNER controls one tank (side chosen at random every episode) and a frozen opponent
controls the other: usually a snapshot from the hall of fame, sometimes the rule-based bot as an anchor.
Opponents run inside the worker (NumPy), so the trainer only batches the learner's observations.

Reward (training only, the arena and referee are untouched) - built to make the agent WANT to win:
  engine reward   +15 per hit dealt, -10 per hit taken, +-100 win / loss, -10 draw   (kept)
  - living bonus  the engine's +0.01/step is removed
  - time penalty  -0.02 per step, so stalling is never free
  - draw penalty  extra -40 on a draw / timeout without a decisive result
  + shaping       gamma*phi(s') - phi(s) with phi = -(aim error + distance to enemy):
                  potential-based, so it speeds up learning to close in and aim without changing
                  which policy is optimal. Uses only what the tank can observe (fog included).
"""
import math
import multiprocessing as mp
import random

import numpy as np

from tank_env import OFFICIAL_MAX_STEPS, RuleBasedBot, TankEnvironment

from .gru_policy import NumpyGRUPolicy, features

GAMMA = 0.997            # must match the PPO discount (potential-based shaping)
TIME_PENALTY = 0.02
DRAW_EXTRA_PENALTY = 40.0
LIVING_BONUS_IN_ENGINE = 0.01
AIM_WEIGHT = 3.0
DIST_WEIGHT = 3.0
TRAIN_SEED_LIMIT = 5_000_000   # evaluation seeds live far above this (referee: 1000.., trainer: 10_000_000..)


def potential(obs: dict) -> float:
    """Higher = closer to the (last known) enemy and aiming at it."""
    err = obs["turret_angle"] - (obs["body_angle"] + obs["enemy_angle_to"])
    aim_error = abs(math.atan2(math.sin(err), math.cos(err))) / math.pi  # 0 aimed .. 1 backwards
    return -(AIM_WEIGHT * aim_error + DIST_WEIGHT * obs["enemy_dist"])


def shaped_reward(engine_reward: float, outcome: str, phi_before: float, phi_after: float, done: bool) -> float:
    r = engine_reward - LIVING_BONUS_IN_ENGINE - TIME_PENALTY
    r += GAMMA * phi_after - phi_before
    if done and outcome.startswith("draw"):
        r -= DRAW_EXTRA_PENALTY
    return r


class _Slot:
    """One arena: learner vs opponent, auto-reset."""

    def __init__(self, rng: random.Random, pool: list, bot_prob: float, latest_prob: float, max_steps: int):
        self.rng, self.bot_prob, self.latest_prob = rng, bot_prob, latest_prob
        self.env = TankEnvironment(max_steps=max_steps)
        self.pool = pool

    def reset(self):
        o1, o2 = self.env.reset(seed=self.rng.randrange(TRAIN_SEED_LIMIT))
        self.side = self.rng.randrange(2)  # 0: learner is tank 1
        if not self.pool or self.rng.random() < self.bot_prob:
            self.opp, self.opp_kind = RuleBasedBot(), "bot"
        else:
            if self.rng.random() < self.latest_prob:
                params = self.pool[-self.rng.randint(1, min(2, len(self.pool)))]
            else:
                params = self.rng.choice(self.pool)
            self.opp, self.opp_kind = NumpyGRUPolicy(params), "hof"
        self.obs = (o1, o2)
        self.phi = potential(self.obs[self.side])
        return features(self.obs[self.side])

    def step(self, learner_action):
        opp_action = self.opp.act(self.obs[1 - self.side])
        a = (learner_action, opp_action) if self.side == 0 else (opp_action, learner_action)
        o1, o2, rewards, done, info = self.env.step(*a)
        self.obs = (o1, o2)
        mine = self.obs[self.side]
        phi = potential(mine)
        reward = shaped_reward(rewards[self.side], info["outcome"], self.phi, phi, done)
        episode = None
        if done:
            winner = {"tank1_win": 0, "tank2_win": 1}.get(info["outcome"])
            result = "draw" if winner is None else ("win" if winner == self.side else "loss")
            episode = {"result": result, "opp": self.opp_kind, "steps": info["step"]}
            feat = self.reset()
        else:
            self.phi = phi
            feat = features(mine)
        return feat, reward, done, episode


def _worker(conn, worker_id, n_envs, base_seed, bot_prob, latest_prob, max_steps):
    rng = random.Random(base_seed * 1000 + worker_id)
    slots = [_Slot(random.Random(rng.random()), [], bot_prob, latest_prob, max_steps) for _ in range(n_envs)]
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                conn.send(np.stack([s.reset() for s in slots]))
            elif cmd == "pool":
                for s in slots:
                    s.pool = payload
                conn.send(True)
            elif cmd == "step":
                feats, rews, dones, episodes = [], [], [], []
                for s, act in zip(slots, payload):
                    f, r, d, ep = s.step(act.tolist())
                    feats.append(f), rews.append(r), dones.append(d)
                    if ep:
                        episodes.append(ep)
                conn.send((np.stack(feats), np.array(rews, np.float32), np.array(dones, np.float32), episodes))
            elif cmd == "close":
                break
    finally:
        conn.close()


class SelfPlayVecEnv:
    """`workers * envs_per_worker` arenas stepped in lockstep; the trainer batches the learner's inputs."""

    def __init__(self, workers=4, envs_per_worker=8, seed=0, bot_prob=0.2, latest_prob=0.35,
                 max_steps=OFFICIAL_MAX_STEPS):
        ctx = mp.get_context("spawn")  # safe next to CUDA; workers never import torch
        self.n_workers, self.per_worker = workers, envs_per_worker
        self.num_envs = workers * envs_per_worker
        self.conns, self.procs = [], []
        for w in range(workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, daemon=True,
                            args=(child, w, envs_per_worker, seed, bot_prob, latest_prob, max_steps))
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)

    def _all(self, cmd, payloads=None):
        for i, c in enumerate(self.conns):
            c.send((cmd, None if payloads is None else payloads[i]))
        return [c.recv() for c in self.conns]

    def reset(self) -> np.ndarray:
        return np.concatenate(self._all("reset"))

    def set_pool(self, pool: list):
        """Hall of fame: list of parameter dicts (NumpyGRUPolicy format). Empty = bot only."""
        self._all("pool", [pool] * self.n_workers)

    def step(self, actions: np.ndarray):
        """actions: (num_envs, 4) float, already in env units (move clipped to [-1,1], shoot +-1)."""
        chunks = np.split(actions, self.n_workers)
        out = self._all("step", chunks)
        feats = np.concatenate([o[0] for o in out])
        rews = np.concatenate([o[1] for o in out])
        dones = np.concatenate([o[2] for o in out])
        episodes = [e for o in out for e in o[3]]
        return feats, rews, dones, episodes

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
