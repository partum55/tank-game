"""
Self-play on the GPU arena: the learner (random side every episode) vs frozen opponents.

Same recipe as the CPU version (mytrain/selfplay_env.py): hall-of-fame snapshots + the rule-based bot as an
anchor, the same aggressive/shaped reward, auto-reset. Differences that come with doing it all in tensors:
  * only `slots` snapshots are "on stage" at a time (their weights are stacked; every env is assigned one
    slot per episode). A new hall-of-fame snapshot enters the oldest slot, otherwise one slot per refresh
    is re-drawn at random from the hall - an env may meet a different opponent behaviour mid-episode.
  * everything (physics, opponents, reward, stats) stays on the device: no CPU round trip per step.
"""
import random

import numpy as np
import torch

from .gpu_arena import DRAW_TIE, DRAW_TIMEOUT, GpuArena, MapBank, T1_WIN, T2_WIN, wrap, PI
from .gru_policy import FEATURE_DIM, MOVE_NOISE_SCALE, OPTIONAL_KEYS, PARAM_KEYS
from .selfplay_env import DRAW_EXTRA_PENALTY, GAMMA, LIVING_BONUS_IN_ENGINE, STUCK_PENALTY, TIME_PENALTY, AIM_WEIGHT, DIST_WEIGHT

BOT_TARGET_DIST = 0.25


def potential(lead_err, dist):
    return -(AIM_WEIGHT * lead_err.abs() / PI + DIST_WEIGHT * dist)


def bot_action(dist, angle_to, turret, body, visible, lidar):
    """tank_env.bots.RuleBasedBot on tensors (E,) -> (E,4)."""
    turn = (torch.sin(angle_to) * 2.0).clamp(-1, 1)
    thrust = torch.where(dist > BOT_TARGET_DIST + 0.05, 0.8,
                         torch.where(dist < BOT_TARGET_DIST - 0.05, -0.5, 0.2))
    err = torch.sin(body + angle_to - turret)
    turret_turn = (err * 3.0).clamp(-1, 1)
    shoot = torch.where(visible & (err.abs() < 0.25), 1.0, -1.0)
    thrust = torch.where(~visible, torch.full_like(thrust, 0.6), thrust)
    turn = torch.where(~visible & (dist < 0.1), torch.ones_like(turn), turn)
    avoid = ~visible & (lidar[:, 3] < 0.05)
    turn = torch.where(avoid, torch.where(lidar[:, 5] > lidar[:, 1], 1.0, -1.0), turn)
    thrust = torch.where(avoid, torch.full_like(thrust, 0.3), thrust)
    return torch.stack([thrust, turn, turret_turn, shoot], -1)


class GpuSelfPlayEnv:
    def __init__(self, num_envs, bank: MapBank, device, hidden, slots=6, bot_prob=0.2, max_steps=1500):
        self.E, self.device, self.S, self.hidden = num_envs, device, slots, hidden
        self.bot_prob = bot_prob
        self.arena = GpuArena(num_envs, bank, device, max_steps)
        self.slot_w = None                      # dict of stacked tensors (S, ...)
        self.slot_std = torch.zeros(slots, 3, device=device)          # exp(log_std) of each slot's policy
        self.slot_stoch = torch.zeros(slots, dtype=torch.bool, device=device)  # snapshot carries its noise => it samples
        self._next_slot, self._last = 0, None
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)
        self.side = z(num_envs, dt=torch.long)  # learner = tank `side`
        self.opp = z(num_envs, dt=torch.long)   # slot index, S = rule-based bot
        self.opp_h = z(num_envs, hidden)
        self.phi = z(num_envs)
        self.stats = z(7)
        self._ar = torch.arange(num_envs, device=device)

    # ---------------------------------------------------------------- hall of fame on stage
    def _put(self, slot, params):
        t = {k: torch.as_tensor(np.asarray(params[k], np.float32), device=self.device) for k in PARAM_KEYS}
        if t["w_in"].shape[1] != FEATURE_DIM:
            raise ValueError(f"opponent snapshot has {t['w_in'].shape[1]} inputs, the arena provides {FEATURE_DIM}: use a fresh hall of fame")
        if self.slot_w is None:
            self.slot_w = {k: v.unsqueeze(0).repeat(self.S, *[1] * v.dim()).clone() for k, v in t.items()}
        for k, v in t.items():
            self.slot_w[k][slot] = v
        has = "log_std" in params
        self.slot_std[slot] = MOVE_NOISE_SCALE * torch.exp(torch.as_tensor(np.asarray(params["log_std"], np.float32), device=self.device)) if has else 0.0
        self.slot_stoch[slot] = has

    def set_pool(self, pool):
        if self.slot_w is None:
            for s in range(self.S):
                self._put(s, pool[random.randrange(len(pool))])
        else:
            new = pool[-1] if pool[-1] is not self._last else pool[random.randrange(len(pool))]
            self._put(self._next_slot, new)
            self._next_slot = (self._next_slot + 1) % self.S
        self._last = pool[-1]

    # ---------------------------------------------------------------- episodes
    def _pick(self, mask):
        """New side + opponent for the envs in `mask`."""
        side = torch.randint(0, 2, (self.E,), device=self.device)
        slot = torch.randint(0, self.S, (self.E,), device=self.device)
        slot = torch.where(torch.rand(self.E, device=self.device) < self.bot_prob, torch.full_like(slot, self.S), slot)
        self.side = torch.where(mask, side, self.side)
        self.opp = torch.where(mask, slot, self.opp)
        self.opp_h = torch.where(mask[:, None], torch.zeros_like(self.opp_h), self.opp_h)

    def _pick_obs(self, tensor, which):
        """tensor (E,2,...) -> (E,...) of the tank `which` (E,)."""
        idx = which.view(-1, *[1] * (tensor.dim() - 1)).expand(-1, 1, *tensor.shape[2:])
        return tensor.gather(1, idx).squeeze(1)

    def reset(self):
        everything = torch.ones(self.E, dtype=torch.bool, device=self.device)
        self._pick(everything)
        self.arena.reset(everything, self.arena.random_maps())
        self.ob = self.arena.observe()
        self.phi = self._phi(self.ob)
        return self._pick_obs(self.ob.features(), self.side)

    def _phi(self, ob):
        g = lambda t: self._pick_obs(t, self.side)
        return potential(g(ob.lead_err), g(ob.dist))

    # ---------------------------------------------------------------- opponents
    def _opp_action(self, ob):
        which = 1 - self.side
        x = self._pick_obs(ob.features(), which)
        out = torch.zeros(self.E, 4, device=self.device)
        if self.slot_w is not None:
            w, Hd = self.slot_w, self.hidden
            for s in range(self.S):
                e = torch.tanh(x @ w["w_in"][s].T + w["b_in"][s])
                gi = e @ w["w_ih"][s].T + w["b_ih"][s]
                gh = self.opp_h @ w["w_hh"][s].T + w["b_hh"][s]
                r = torch.sigmoid(gi[:, :Hd] + gh[:, :Hd])
                z = torch.sigmoid(gi[:, Hd:2 * Hd] + gh[:, Hd:2 * Hd])
                n = torch.tanh(gi[:, 2 * Hd:] + r * gh[:, 2 * Hd:])
                h_new = (1 - z) * n + z * self.opp_h
                o = h_new @ w["w_out"][s].T + w["b_out"][s]
                m = (self.opp == s).unsqueeze(1)
                self.opp_h = torch.where(m, h_new, self.opp_h)
                out = torch.where(m, o, out)
        slot = self.opp.clamp(max=self.S - 1)
        stoch = self.slot_stoch[slot].unsqueeze(1)
        # same behaviour as the learner it was trained as: Gaussian moves, Bernoulli shot (deterministic if the snapshot has no noise)
        move = (out[:, :3] + stoch * self.slot_std[slot] * torch.randn_like(out[:, :3])).clamp(-1, 1)
        shoot = torch.where(stoch, torch.where(torch.rand_like(out[:, 3:4]) < torch.sigmoid(out[:, 3:4]), 1.0, -1.0),
                            torch.where(out[:, 3:4] > 0, 1.0, -1.0))
        net_act = torch.cat([move, shoot], 1)
        g = lambda t: self._pick_obs(t, which)
        bot = bot_action(g(ob.dist), g(ob.angle_to), g(ob.turret), g(ob.body), g(ob.visible), g(ob.lidar))
        return torch.where((self.opp == self.S).unsqueeze(1), bot, net_act)

    # ---------------------------------------------------------------- step
    def step(self, act):
        """act (E,4): learner action (move in [-1,1], shoot +-1). Returns (obs (E,31), reward (E,), done (E,))."""
        opp_act = self._opp_action(self.ob)
        mine0 = (self.side == 0).unsqueeze(1)
        both = torch.stack([torch.where(mine0, act, opp_act), torch.where(mine0, opp_act, act)], 1)
        r_eng, done, outcome = self.arena.tick(both)

        post = self.arena.observe(lidar=False)  # enough for the shaping potential
        phi = self._phi(post)
        stuck = self._pick_obs(self.arena.blocked_move.float(), self.side)
        reward = (self._pick_obs(r_eng, self.side) - LIVING_BONUS_IN_ENGINE - TIME_PENALTY - STUCK_PENALTY * stuck
                  + GAMMA * phi - self.phi)
        draw = (outcome == DRAW_TIE) | (outcome == DRAW_TIMEOUT)
        reward = reward - DRAW_EXTRA_PENALTY * draw

        win = ((outcome == T1_WIN) & (self.side == 0)) | ((outcome == T2_WIN) & (self.side == 1))
        loss = done & ~win & ~draw
        is_bot = self.opp == self.S
        d = done.float()
        self.stats += torch.stack([d.sum(), (d * self.arena.steps).sum(), (win.float()).sum(), loss.float().sum(),
                                   draw.float().sum(), (d * is_bot).sum(), ((win & is_bot).float()).sum()])

        self._pick(done)
        self.arena.reset(done, self.arena.random_maps())
        self.ob = self.arena.observe()
        self.phi = torch.where(done, self._phi(self.ob), phi)
        return self._pick_obs(self.ob.features(), self.side), reward, d

    def pop_stats(self):
        s = self.stats.cpu().tolist()
        self.stats.zero_()
        return dict(n=s[0], len=s[1], win=s[2], loss=s[3], draw=s[4], bot_n=s[5], bot_win=s[6])

    def close(self):
        pass
