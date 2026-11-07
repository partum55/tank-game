import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import train_ppo
from mytrain.gpu_arena import MapBank
from mytrain.gpu_selfplay import GpuSelfPlayEnv
from mytrain.gru_policy import FEATURE_DIM, random_params
from mytrain.mapmix import build_mix
from mytrain.rollout import Rollout


def make(static, seed=0, E=24, T=20):
    random.seed(seed)            # hall-of-fame slots are filled with Python's random
    torch.manual_seed(seed)
    bank = MapBank.from_rust(list(range(10)), "cpu", build_mix(10, 1))
    env = GpuSelfPlayEnv(E, bank, "cpu", 16, slots=2, bot_prob=0.3, max_steps=15)  # short episodes: resets happen inside the rollout
    pool = [dict(random_params(FEATURE_DIM, 16, 16, s), log_std=np.full(3, -0.7, np.float32)) for s in (1, 2)]
    env.set_pool(pool)
    net = train_ppo.Net(FEATURE_DIM, 16, 16)
    torch.manual_seed(seed + 100)
    return Rollout(net, env, T, 16, torch.device("cpu"), 0.05, static_env=static), env


def test_rollout_buffers_are_consistent():
    ro, env = make(static=False)
    ro.run()
    assert int(ro.t) == ro.T
    for buf in (ro.obs_buf, ro.rew_buf, ro.val_buf, ro.logp_buf, ro.move_buf):
        assert torch.isfinite(buf).all()
    # the observation at t+1 is the one the env returned at t; `first` marks the step after a finished episode
    assert torch.equal(ro.first_buf[1:], ro.done_buf[:-1])
    assert ro.done_buf.sum() > 0, "15-step episodes must end inside a 20-step rollout"
    last_done = ro.done_buf[-1].clone()
    ro.run()                                                # a second rollout continues the same games
    assert torch.equal(ro.first_buf[0], last_done)


def test_static_state_step_gives_bit_identical_trajectories():
    """The identity-preserving step (what a CUDA graph replays) must not change the game."""
    a, env_a = make(static=False)
    b, env_b = make(static=True)
    for _ in range(4):
        torch.manual_seed(7)
        a.run()
        torch.manual_seed(7)
        b.run()
        for name in ("obs_buf", "move_buf", "first_buf", "shoot_buf", "logp_buf", "val_buf", "rew_buf", "done_buf"):
            assert torch.equal(getattr(a, name), getattr(b, name)), name
        assert torch.equal(env_a.arena.pos, env_b.arena.pos) and torch.equal(env_a.phi, env_b.phi)
    # and the state tensors really kept their identity
    pos_id = env_b.arena.pos.data_ptr()
    b.run()
    assert env_b.arena.pos.data_ptr() == pos_id


def test_graph_request_falls_back_cleanly_without_cuda():
    torch.manual_seed(0)
    bank = MapBank.from_rust(list(range(5)), "cpu")
    env = GpuSelfPlayEnv(8, bank, "cpu", 16, slots=1)
    env.set_pool([random_params(FEATURE_DIM, 16, 16, 1)])
    ro = Rollout(train_ppo.Net(FEATURE_DIM, 16, 16), env, 8, 16, torch.device("cpu"), 0.05, graph=True)
    assert ro.graph is None and "OFF" in ro.graph_note and not ro.static_env
    ro.run()
    assert int(ro.t) == 8 and torch.isfinite(ro.obs_buf).all()
