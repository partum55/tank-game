import math
import sys
from pathlib import Path

import numpy as np
import pytest

from tank_env import (
    MLPAgent,
    LidarObservationAdapter,
    SimpleMLP,
    TankEnvironment,
    VectorObservationAdapter,
    load_agent,
    sanitize_action,
)
import referee
import train_ga


def test_sanitize_action_is_a_trust_boundary():
    assert sanitize_action([2.0, -5.0, 0.3, 1.0]) == [1.0, -1.0, 0.3, 1.0]
    assert sanitize_action([float("nan"), math.inf, -math.inf, 0.9]) == [0.0, 0.0, 0.0, 0.9]
    assert sanitize_action([0.5]) == [0.5, 0.0, 0.0, 0.0]
    assert sanitize_action([1, 2, 3, 4, 5]) == [1.0, 1.0, 1.0, 1.0]
    assert sanitize_action(np.array([0.1, 0.2, 0.3, 0.4])) == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert sanitize_action(None) == [0.0] * 4
    assert sanitize_action(["x", None]) == [0.0] * 4


def test_nan_action_cannot_poison_the_physics():
    env = TankEnvironment(max_steps=50)
    env.reset(seed=1)
    for _ in range(5):
        obs1, obs2, *_ = env.step([float("nan")] * 4, [math.inf, 0, 0, 0])
    for obs in (obs1, obs2):
        assert all(math.isfinite(obs[k]) for k in ("x", "y", "speed", "body_angle", "hp"))


def test_load_agent_from_file_with_sibling_module(tmp_path):
    (tmp_path / "helper_mod_xyz.py").write_text("VALUE = 0.25\n")
    (tmp_path / "my_agent.py").write_text(
        "from helper_mod_xyz import VALUE\n"
        "class A:\n"
        "    def __init__(self): self.resets = 0\n"
        "    def reset(self): self.resets += 1\n"
        "    def act(self, obs): return [VALUE, 0, 0, -1]\n"
        "def make_agent(): return A()\n"
    )
    agent = load_agent(str(tmp_path / "my_agent.py"))
    assert agent.act({}) == [0.25, 0, 0, -1]
    assert str(tmp_path) not in sys.path  # sys.path is restored


def test_load_agent_errors(tmp_path):
    with pytest.raises(ValueError):
        load_agent("model.npz")
    with pytest.raises(FileNotFoundError):
        load_agent(str(tmp_path / "missing.py"))
    (tmp_path / "no_factory.py").write_text("x = 1\n")
    with pytest.raises(ValueError):
        load_agent(str(tmp_path / "no_factory.py"))
    (tmp_path / "bad_agent.py").write_text("def make_agent(): return object()\n")
    with pytest.raises(ValueError):
        load_agent(str(tmp_path / "bad_agent.py"))


def test_mlp_agent_checks_dimensions():
    adapter = VectorObservationAdapter()
    with pytest.raises(ValueError):
        MLPAgent(SimpleMLP(adapter.output_dim + 1, [4], 4), adapter)


def test_env_seed_reproducible_and_map_symmetric():
    env = TankEnvironment(max_steps=50)
    a = env.reset(seed=123)
    b = env.reset(seed=123)
    assert a == b
    obs1 = a[0]
    # tank 2 sees the same map from the point-mirrored side
    assert a[1]["obstacles"] == obs1["obstacles"]
    assert math.isclose(a[0]["x"] + a[1]["x"], 1.0, abs_tol=1e-5)
    assert math.isclose(a[0]["y"] + a[1]["y"], 1.0, abs_tol=1e-5)


def test_play_match_and_series_swap_sides():
    a, b = load_agent("rule_based"), load_agent("random")
    r = referee.play_match(a, b, seed=5, max_steps=80)
    assert r["winner"] in (0, 1, 2) and r["steps"] <= 80
    res = referee.play_series(a, b, seeds=[5], max_steps=80)
    assert res["games"] == 2
    for n in "AB":
        s = res[n]
        assert s["wins"] + s["losses"] + s["draws"] == 2
    assert res["A"]["wins"] == res["B"]["losses"] and res["A"]["draws"] == res["B"]["draws"]


def test_ga_one_generation_writes_a_playable_agent(tmp_path):
    adapter = LidarObservationAdapter()
    env = TankEnvironment(max_steps=60)
    ga = train_ga.GAPopulation(4, adapter.output_dim, [8, 8], 4)
    opponent = load_agent("rule_based")
    fits = [train_ga.evaluate_individual(m, adapter, opponent, env, [1, 2]) for m in ga.population]
    best, best_fit, _ = ga.evolve(fits)
    best.save(str(tmp_path / "best_model.npz"))
    train_ga.write_agent_file(str(tmp_path), LidarObservationAdapter)

    agent = load_agent(str(tmp_path / "agent.py"))
    obs, _ = env.reset(seed=9)
    action = sanitize_action(agent.act(obs))
    assert len(action) == 4
    res = referee.play_series(agent, load_agent("rule_based"), seeds=[3], max_steps=60)
    assert res["games"] == 2


# ---- agents in their own processes (both models run on one laptop) ----

from tank_env.remote import AgentFailure, RemoteAgent


def _agent_file(folder, body):
    folder.mkdir(exist_ok=True)
    path = folder / "agent.py"
    path.write_text(body)
    return str(path)


def test_remote_agent_isolates_same_named_modules_and_stdout(tmp_path):
    """Two players, both with a sibling `helper.py`: no clash. print() can't break the pipe."""
    mk = lambda v: (
        "from helper import VALUE\n"
        "def make_agent():\n"
        "    class A:\n"
        "        def act(self, obs):\n"
        "            print('chatty agent')\n"
        f"            return [VALUE, float('nan'), 0, 0]\n"
        "    return A()\n"
    )
    for name, v in (("a", 0.25), ("b", 0.75)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "helper.py").write_text(f"VALUE = {v}\n")
    pa = _agent_file(tmp_path / "a", mk(0))
    pb = _agent_file(tmp_path / "b", mk(0))
    ra, rb = RemoteAgent(pa), RemoteAgent(pb)
    try:
        assert ra.act({"x": 1}) == [0.25, 0.0, 0.0, 0.0]  # NaN sanitized in the agent process
        assert rb.act({"x": 1}) == [0.75, 0.0, 0.0, 0.0]
        ra.reset()
    finally:
        ra.close(), rb.close()


def test_remote_agent_failures(tmp_path):
    crash = _agent_file(tmp_path / "c", "def make_agent():\n"
                        "    class A:\n        def act(self, o): raise RuntimeError('boom')\n    return A()\n")
    r = RemoteAgent(crash)
    with pytest.raises(AgentFailure, match="boom"):
        r.act({})
    r.close()

    slow = _agent_file(tmp_path / "s", "import time\ndef make_agent():\n"
                       "    class A:\n        def act(self, o):\n            time.sleep(2)\n            return [0]*4\n    return A()\n")
    r = RemoteAgent(slow, timeout_ms=150)
    with pytest.raises(AgentFailure, match="no answer"):
        r.act({})
    r.close()

    broken = _agent_file(tmp_path / "b", "raise ImportError('missing dependency')\n")
    with pytest.raises(AgentFailure, match="missing dependency"):
        RemoteAgent(broken)


def test_referee_forfeit_on_crash_then_restart(tmp_path):
    flaky = _agent_file(tmp_path / "f", "def make_agent():\n"
                        "    class A:\n        def act(self, o): raise RuntimeError('boom')\n    return A()\n")
    bad = RemoteAgent(flaky)
    good = load_agent("rule_based")
    try:
        r = referee.play_match(bad, good, seed=3, max_steps=50)
        assert r["outcome"] == "forfeit_tank1" and r["winner"] == 2 and r["failed"] == 1
        res = referee.play_series(bad, good, seeds=[3], max_steps=50)
        assert res["A"]["forfeits"] == 2 and res["A"]["losses"] == 2 and res["B"]["wins"] == 2
    finally:
        bad.close()


def test_remote_agent_plays_a_real_match(tmp_path):
    path = _agent_file(tmp_path / "t", "from tank_env import RuleBasedBot\ndef make_agent(): return RuleBasedBot()\n")
    remote = RemoteAgent(path)
    try:
        r = referee.play_match(remote, load_agent("random"), seed=11, max_steps=100)
        assert r["failed"] is None and r["steps"] <= 100
    finally:
        remote.close()
