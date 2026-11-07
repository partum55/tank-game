"""GPU self-play pieces that the arena parity test does not cover: the bot port, the in-tensor GRU opponent, bookkeeping."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tank_env import RuleBasedBot, TankEnvironment
from mytrain.gpu_arena import GpuArena, MapBank
from mytrain.gpu_selfplay import GpuSelfPlayEnv, bot_action
from mytrain.gru_policy import FEATURE_DIM, NumpyGRUPolicy, random_params
from tests.test_gpu_parity import random_actions


def test_bot_port_matches_the_python_bot():
    seeds = list(range(300, 312))
    bank = MapBank.from_rust(seeds, "cpu")
    arena = GpuArena(len(seeds), bank, "cpu", max_steps=500)
    arena.reset(torch.ones(len(seeds), dtype=torch.bool), torch.arange(len(seeds)))
    envs = [TankEnvironment(max_steps=500) for _ in seeds]
    rust = [e.reset(seed=s) for e, s in zip(envs, seeds)]
    rng = np.random.default_rng(1)
    acts = np.stack([random_actions(rng, 250) for _ in seeds], 1)
    bot, mism, checked = RuleBasedBot(), 0, 0
    for t in range(250):
        ob = arena.observe()
        for obs_i in (0, 1):
            g = lambda x: x[:, obs_i]
            got = bot_action(g(ob.dist), g(ob.angle_to), g(ob.turret), g(ob.body), g(ob.visible), g(ob.lidar)).numpy()
            for e in range(len(seeds)):
                want = np.array(bot.act(rust[e][obs_i]))
                checked += 1
                # tolerance for float32-vs-float64 trig near thresholds; the shoot flag is a hard +-1
                mism += not (np.abs(got[e, :3] - want[:3]).max() < 2e-3 and got[e, 3] == want[3])
        arena.tick(torch.as_tensor(acts[t]))
        rust = [(lambda o: (o[0], o[1]))(env.step(acts[t, e, 0], acts[t, e, 1])) for e, env in enumerate(envs)]
    assert mism / checked < 0.002, f"{mism} of {checked} bot decisions differ"


def test_in_tensor_gru_opponent_matches_the_numpy_policy():
    E, hidden = 4, 16
    params = random_params(FEATURE_DIM, hidden, hidden, seed=3)
    bank = MapBank.from_rust(list(range(50, 60)), "cpu")
    env = GpuSelfPlayEnv(E, bank, "cpu", hidden, slots=2, bot_prob=0.0, max_steps=10_000)
    env.set_pool([params])
    env.reset()
    env.side[:] = 0                      # learner = tank 1, the opponent is tank 2
    env.opp[:] = 1                       # every env plays slot 1
    env.opp_h.zero_()
    ref = [NumpyGRUPolicy(params) for _ in range(E)]
    rng = np.random.default_rng(0)
    for _ in range(40):
        feats = env._pick_obs(env.ob.features(), 1 - env.side).numpy()
        h_before = env.opp_h.clone()
        got = env._opp_action(env.ob).numpy()
        env.opp_h = h_before             # the real env.step() advances the opponent's memory itself
        want = []
        for e, r in enumerate(ref):
            out = r.forward(feats[e])
            want.append([*np.clip(out[:3], -1, 1), 1.0 if out[3] > 0 else -1.0])
        want = np.array(want)
        assert np.abs(got - want).max() < 1e-4
        act = torch.as_tensor(rng.uniform(-1, 1, (E, 4)).astype(np.float32))
        env.step(act)
        # keep the comparison valid across an episode reset: the env wipes the opponent's memory
        for e in range(E):
            if env.arena.steps[e] == 0:
                ref[e].reset()


def test_stats_count_finished_episodes_and_rewards_are_finite():
    bank = MapBank.from_rust(list(range(70, 80)), "cpu")
    env = GpuSelfPlayEnv(32, bank, "cpu", 8, slots=2, bot_prob=0.5, max_steps=40)
    env.set_pool([random_params(FEATURE_DIM, 8, 8, 1), random_params(FEATURE_DIM, 8, 8, 2)])
    env.reset()
    for _ in range(100):
        act = torch.rand(32, 4) * 2 - 1
        act[:, 3] = torch.where(act[:, 3] > 0, 1.0, -1.0)
        obs, rew, done = env.step(act)
        assert obs.shape == (32, FEATURE_DIM) and torch.isfinite(obs).all() and torch.isfinite(rew).all()
    st = env.pop_stats()
    assert st["n"] >= 32 and abs(st["win"] + st["loss"] + st["draw"] - st["n"]) < 1e-6
    assert 0 < st["bot_n"] < st["n"]
    assert env.pop_stats()["n"] == 0
