"""Pre-match adaptation (mytrain/adapt.py): the opponent's view and actions are rebuilt from our telemetry correctly, learning is bounded and deterministic."""
import time

import numpy as np
import pytest

import referee
from tank_env import RuleBasedBot, TankEnvironment, load_agent
from mytrain.adapt import HEAD_MAX_REL, enemy_actions, enemy_inputs, maybe_blocked, visible_runs
from mytrain.gru_policy import FEATURE_DIM, SHOT_THETA, FeatureExtractor, NumpyGRUPolicy, random_params, write_standalone_agent

SEEDS = (1002, 1005, 1010, 1013)    # maps on which the two rule-based bots see each other


def make_policy(adapt="move,shot,bc,rl", seed=1, hidden=16):
    params = dict(random_params(FEATURE_DIM, hidden, hidden, seed), log_std=np.full(3, -0.7, np.float32))
    return NumpyGRUPolicy(params, adapt=adapt)


def play_bot_match(pol, seed, ticks=600):
    """Our tank is driven by a rule-based bot but its observations go through the policy's recorder; the opponent is another bot.
    Returns the opponent's TRUE network inputs and actions per tick, to compare with what the adapter reconstructs."""
    env = TankEnvironment(max_steps=ticks)
    o1, o2 = env.reset(seed=seed)
    bot1, bot2, fx_true = RuleBasedBot(), RuleBasedBot(), FeatureExtractor()
    true_inputs, true_actions = [], []
    for _ in range(ticks):
        out = pol.forward(pol.fx(o1))
        a1, a2 = bot1.act(o1), bot2.act(o2)
        pol.adapter.record(o1, pol.fx, pol.h, out, out[:3], a1[3], False)
        true_inputs.append(fx_true(o2))
        true_actions.append(a2)
        o1, o2, _, done, _ = env.step(a1, a2)
        if done:
            break
    return np.array(true_inputs), np.array(true_actions)


def test_the_opponents_inputs_are_rebuilt_from_our_telemetry():
    errs, n = [], 0
    for seed in SEEDS:
        pol = make_policy()
        true_inputs, _ = play_bot_match(pol, seed)
        rows = pol.adapter.rows
        runs = visible_runs(rows)
        starts = {i0 for i0, _ in runs}
        pairs = enemy_inputs(rows, runs, pol.adapter.rects)
        n += len(pairs)
        for k, x in pairs:
            e = np.abs(x - true_inputs[k])
            if k in starts or maybe_blocked(rows, k, pol.adapter.rects):
                e[:3] = 0.0              # its own velocity is unknown on the first tick of a stretch and when it pushes against a wall
            errs.append(e)
    assert n > 150, "the bots hardly saw each other: pick other seeds"
    errs = np.array(errs)
    # every one of the 36 inputs matches what the opponent's own agent computed, up to the speed estimate (collisions) and float noise
    worst = np.percentile(errs, 99, axis=0)
    assert (worst < 0.05).all(), f"columns off: {np.nonzero(worst >= 0.05)[0].tolist()} {np.round(worst[worst >= 0.05], 3).tolist()}"


def test_the_opponents_actions_are_rebuilt_from_our_telemetry():
    diff, known, shots_rebuilt, shots_real = [], [], 0, 0
    for seed in SEEDS:
        pol = make_policy()
        _, true_actions = play_bot_match(pol, seed)
        rows, rects = pol.adapter.rows, pol.adapter.rects
        for i0, i1 in visible_runs(rows):
            for k in range(i0 + 1, i1 - 1):
                got = enemy_actions(rows, k, rects)
                want = true_actions[k]
                diff.append(np.abs(np.array(got[:3]) - np.array(want[:3])))
                known.append(got[4])
                shots_rebuilt += got[3]
                shots_real += want[3] > 0 and got[3] > 0
    diff, known = np.array(diff), np.array(known)
    assert len(diff) > 100 and known.sum() > 50
    assert np.percentile(diff[:, 1:], 99, axis=0).max() < 0.01          # turn and turret turn are exact (angle differences)
    assert diff[known, 0].max() < 0.02                                    # thrust is exact wherever it is declared readable ...
    assert (diff[~known, 0] > 0.05).any()                                 # ... and the declared-unreadable ticks include the real blocked ones
    assert shots_rebuilt > 0 and shots_real == shots_rebuilt              # every reconstructed shot was a commanded shot


def test_adaptation_is_bounded_deterministic_and_changes_the_agent():
    def run():
        pol = make_policy()
        w0 = pol.p["w_out"].copy()
        for seed in SEEDS:
            play_bot_match(pol, seed)
            pol.reset()                                                  # learns from the match that just ended
        return pol, w0

    a, w0 = run()
    b, _ = run()
    assert np.array_equal(a.p["w_out"], b.p["w_out"]) and np.array_equal(a.p["b_out"], b.p["b_out"])   # deterministic
    assert not np.array_equal(a.p["w_out"], w0), "the output layer did not move"
    start = np.hstack([w0, make_policy().p["b_out"][:, None]])
    now = np.hstack([a.p["w_out"], a.p["b_out"][:, None]])
    assert np.linalg.norm(now - start) <= HEAD_MAX_REL * np.linalg.norm(start) * 1.001      # never far from the exported layer
    # the other parts were fitted too: a refit shot model (prior = the defaults) and a decision on the motion model
    assert any("theta" in str(h.get("shot")) for h in a.adapter.history)
    assert not np.allclose(a.fx.theta, SHOT_THETA)
    assert all(h["head"] for h in a.adapter.history)
    # the network keeps its inputs and the unchanged parts of its weights
    assert np.array_equal(a.p["w_in"], b.p["w_in"]) and np.array_equal(a.p["w_hh"], b.p["w_hh"])


def test_parts_can_be_switched_off_and_a_plain_agent_does_not_adapt():
    plain = make_policy(adapt=None)
    assert plain.adapter is None
    only_move = make_policy(adapt="move")
    w0 = only_move.p["w_out"].copy()
    for seed in SEEDS[:2]:
        play_bot_match(only_move, seed)
        only_move.reset()
    assert np.array_equal(only_move.p["w_out"], w0) and np.allclose(only_move.fx.theta, SHOT_THETA)   # head and shot model untouched
    with pytest.raises(ValueError):
        make_policy(adapt="move,telepathy")


def test_motion_model_learns_a_circling_enemy_and_keeps_constant_velocity_for_a_straight_one():
    def rows_for(path):
        return [dict(epos=p) for p in path]

    t = np.arange(500)
    circle = np.stack([600 + 120 * np.cos(0.035 * t), 400 + 120 * np.sin(0.035 * t)], 1)       # ~4.2 px/tick on a circle
    straight = np.stack([100 + 2.0 * t, 300 + 0.0 * t], 1)
    pol = make_policy()
    note = pol.adapter._fit_move(rows_for(circle), [(0, 500)])
    assert note.startswith("model on") and pol.fx.vel_model is not None
    A, b = pol.fx.vel_model
    v = (circle[201] - circle[200]) / 4.5                                                          # velocity now, in units of MAX_SPEED
    future = (circle[226] - circle[201]) / 25 / 4.5                                                # what it really does over the next 25 ticks
    assert np.linalg.norm(A @ v + b - future) < 0.5 * np.linalg.norm(v - future)                   # clearly better than "keeps its velocity"

    pol = make_policy()
    assert pol.adapter._fit_move(rows_for(straight), [(0, 500)]).startswith("kept constant velocity")
    assert pol.fx.vel_model is None


def test_adaptation_fits_in_the_referee_time_limit():
    pol = make_policy(hidden=256)
    play_bot_match(pol, SEEDS[0], ticks=400)
    rows = len(pol.adapter.rows)
    t0 = time.perf_counter()
    pol.reset()
    per_row = (time.perf_counter() - t0) / rows
    assert per_row * 1500 < 3.0, f"{per_row * 1e3:.2f} ms per recorded tick: a 1500-tick match would need {per_row * 1500:.1f} s (limit 5 s)"


def test_exported_adaptive_agent_runs_standalone_in_the_referee(tmp_path):
    params = dict(random_params(FEATURE_DIM, 16, 16, 7), log_std=np.full(3, -0.7, np.float32))
    write_standalone_agent(str(tmp_path), params, adapt="move,shot,bc,rl")
    assert (tmp_path / "tank_adapt.py").exists()
    agent = str(tmp_path / "agent.py")
    # the series reuses one process, so reset() before matches 2+ really adapts inside the isolated agent process
    stats = referee.play_series(referee.make_agent(agent), referee.make_agent("rule_based"), [1002, 1005], max_steps=300)
    assert stats["games"] == 4 and stats["A"]["forfeits"] == 0
    assert load_agent(agent).adapter is not None
