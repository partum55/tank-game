import math

import numpy as np
import pytest

import referee
from tank_env import VectorObservationAdapter, load_agent
from mytrain.gru_policy import NumpyGRUPolicy, PARAM_KEYS, features, random_params, write_standalone_agent
from mytrain.selfplay_env import (
    DRAW_EXTRA_PENALTY,
    GAMMA,
    SelfPlayVecEnv,
    TIME_PENALTY,
    potential,
    shaped_reward,
)

OBS = dict(turret_angle=0.0, body_angle=0.0, enemy_angle_to=0.0, enemy_dist=0.5)


def test_gru_policy_acts_deterministically_and_remembers():
    params = random_params(27, 16, 16, seed=1)
    pol = NumpyGRUPolicy(params)
    x = np.linspace(-1, 1, 27).astype(np.float32)
    a, b = pol.forward(x).copy(), pol.forward(x).copy()
    assert not np.allclose(a, b), "the hidden state must change the output (memory)"
    pol.reset()
    assert np.allclose(pol.forward(x), a), "reset() must wipe the memory"

    action = pol.act({})  # missing fields fall back to defaults in the adapter
    assert len(action) == 4 and all(-1 <= v <= 1 for v in action) and action[3] in (-1.0, 1.0)


def test_gru_policy_save_load_and_dimension_check(tmp_path):
    params = random_params(27, 8, 8, seed=2)
    pol = NumpyGRUPolicy(params)
    pol.save(str(tmp_path / "p.npz"))
    other = NumpyGRUPolicy.load(str(tmp_path / "p.npz"))
    x = np.ones(27, np.float32)
    assert np.allclose(pol.forward(x), other.forward(x))
    with pytest.raises(ValueError):
        NumpyGRUPolicy(random_params(10, 8, 8))


def test_reward_is_aggressive_and_potential_based():
    # closer and better aimed = higher potential
    far_unaimed = dict(OBS, enemy_dist=0.9, enemy_angle_to=math.pi)
    assert potential(OBS) > potential(far_unaimed)
    # standing still with no progress costs time every step (no free stalling)
    idle = shaped_reward(0.01, "in_progress", -1.0, -1.0, False)  # engine's +0.01 living bonus is removed
    assert idle == pytest.approx(-TIME_PENALTY - (GAMMA - 1.0), abs=1e-6) and idle < 0
    # a timeout draw is clearly worse than the engine's own draw penalty
    draw = shaped_reward(-10.0, "draw_timeout", -1.0, -1.0, True)
    no_draw = shaped_reward(-10.0, "in_progress", -1.0, -1.0, True)
    assert draw == pytest.approx(no_draw - DRAW_EXTRA_PENALTY)
    # winning beats everything; getting hit costs less than landing a hit gains
    assert shaped_reward(100.0, "tank1_win", -1, -1, True) > shaped_reward(15.0, "in_progress", -1, -1, False)
    assert shaped_reward(15.0 - 10.0, "in_progress", -1, -1, False) > 0


def test_vec_env_steps_resets_and_reports_episodes():
    pool = [random_params(27, 8, 8, seed=s) for s in range(2)]
    env = SelfPlayVecEnv(workers=2, envs_per_worker=2, seed=3, max_steps=30)
    try:
        env.set_pool(pool)
        obs = env.reset()
        assert obs.shape == (4, 27) and np.isfinite(obs).all()
        finished = []
        for _ in range(70):
            act = np.random.uniform(-1, 1, (4, 4))
            act[:, 3] = np.where(act[:, 3] > 0, 1.0, -1.0)
            obs, rew, done, eps = env.step(act)
            assert obs.shape == (4, 27) and rew.shape == (4,) and done.shape == (4,)
            assert np.isfinite(obs).all() and np.isfinite(rew).all()
            finished += eps
        assert finished, "30-step episodes must have finished"
        assert all(e["result"] in ("win", "loss", "draw") and e["opp"] in ("bot", "hof") for e in finished)
        assert {e["opp"] for e in finished} <= {"bot", "hof"}
        env.set_pool([])  # empty pool = rule-based bot only
        for _ in range(35):
            _, _, _, eps = env.step(np.zeros((4, 4)))
        assert all(e["opp"] == "bot" for e in eps) or not eps
    finally:
        env.close()


def test_features_match_the_shared_adapter():
    from tank_env import TankEnvironment

    o1, _ = TankEnvironment(max_steps=20).reset(seed=4)
    assert np.allclose(features(o1), VectorObservationAdapter().extract(o1), atol=1e-6)


def test_exported_agent_is_standalone_and_plays_in_the_referee(tmp_path):
    write_standalone_agent(str(tmp_path), random_params(27, 16, 16, 5))
    assert (tmp_path / "tank_gru_policy.py").exists()
    agent = load_agent(str(tmp_path / "agent.py"))
    res = referee.play_series(agent, load_agent("rule_based"), seeds=[7], max_steps=60)
    assert res["games"] == 2 and res["A"]["forfeits"] == 0


# ---- the torch side (skipped where PyTorch is not installed, e.g. the laptop duel venv) ----

def test_torch_network_matches_numpy_policy():
    pytest.importorskip("torch")
    import train_ppo

    assert train_ppo.self_check() < 1e-4


def test_torch_export_roundtrip_and_write_export(tmp_path):
    pytest.importorskip("torch")
    import train_ppo

    net = train_ppo.Net(27, 16, 16)
    params = train_ppo.export_params(net)
    assert set(params) == set(PARAM_KEYS)
    train_ppo.write_export(str(tmp_path), params)  # same files a Colab run exports
    agent = load_agent(str(tmp_path / "agent.py"))
    r = referee.play_match(agent, load_agent("rule_based"), seed=11, max_steps=50)
    assert r["failed"] is None
