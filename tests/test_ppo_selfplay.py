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


def test_bundle_contains_what_a_reviewer_needs(tmp_path):
    import argparse, json, os
    from mytrain.bundle import append_metrics, make_bundle, write_config

    run = tmp_path / "run"
    (run / "hof").mkdir(parents=True)
    for u in range(10, 101, 10):  # 10 snapshots -> bundle keeps 6 evenly spaced
        NumpyGRUPolicy(random_params(27, 8, 8, u)).save(str(run / "hof" / f"snap_{u:05d}.npz"))
    write_standalone_agent(str(run / "export_best"), random_params(27, 8, 8, 1))
    write_standalone_agent(str(run / "export_latest"), random_params(27, 8, 8, 2))
    write_config(str(run), argparse.Namespace(hidden=8, seed=1), {"num_envs": 4})
    write_config(str(run), argparse.Namespace(hidden=8, seed=1), {"num_envs": 4})  # resumed session
    append_metrics(str(run), dict(update=1, steps=100, win=0.5, eval_vs_bot=0.4))
    (run / "train.log").write_text("\n".join(f"line {i}" for i in range(1000)))
    out = make_bundle(str(run))
    names = set(os.listdir(out))
    assert {"config.json", "metrics.csv", "export_best", "export_latest", "hof"} <= names
    snaps = sorted(os.listdir(os.path.join(out, "hof")))
    assert len(snaps) == 6 and snaps[0] == "snap_00010.npz" and snaps[-1] == "snap_00100.npz"
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert cfg["rules_version"] and len(cfg["runs"]) == 2
    assert load_agent(os.path.join(out, "export_best", "agent.py")).act({})  # the bundled agent runs
    log = open(os.path.join(out, "train_log.txt")).read().splitlines()
    assert log[0] == "line 0" and log[-1] == "line 999" and any("trimmed" in l for l in log) and len(log) < 400


def test_fused_unroll_matches_the_step_by_step_reference():
    torch = pytest.importorskip("torch")
    import train_ppo

    torch.manual_seed(1)
    net = train_ppo.Net(27, 16, 16)
    for p in net.parameters():
        p.data += 0.2 * torch.randn_like(p)
    T, B = 30, 7
    x = torch.randn(T, B, 27)
    h0 = torch.randn(B, 16)
    first = (torch.rand(T, B) < 0.08).float()
    first[0, :3] = 1.0          # window opens an episode for some envs only
    first[:, 6] = 0.0           # an env with no reset at all
    first[5, 5] = first[6, 5] = 1.0  # back-to-back resets
    results = []
    for fn in (net.unroll_reference, net.unroll):
        net.zero_grad()
        out, val = fn(x, h0, first)
        (out.pow(2).sum() + val.pow(2).sum()).backward()
        results.append((out.detach(), val.detach(), [p.grad.clone() for p in net.parameters() if p.grad is not None]))
    (o1, v1, g1), (o2, v2, g2) = results
    assert torch.allclose(o1, o2, atol=1e-5) and torch.allclose(v1, v2, atol=1e-5)
    assert all(torch.allclose(a, b, atol=1e-4) for a, b in zip(g1, g2))


def test_autotune_picks_a_size_from_a_monotone_benchmark():
    torch = pytest.importorskip("torch")
    import train_ppo
    from mytrain.gpu_arena import MapBank

    bank = MapBank.from_rust(list(range(20)), "cpu")
    table, chosen = train_ppo.autotune_envs(bank, torch.device("cpu"), 16, 2, max_envs=256, floor=64)
    sizes = [e for e, _ in table]
    assert sizes == sorted(sizes) and sizes[0] == 64 and chosen in sizes
    assert all(sps > 0 for _, sps in table)
