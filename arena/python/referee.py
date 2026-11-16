#!/usr/bin/env python3
"""
Referee: the only official way to compare two agents.

    PYTHONPATH=python python3 python/referee.py my_agent.py friend_agent.py --seeds 20
    PYTHONPATH=python python3 python/referee.py my_agent.py rule_based --seeds 1 --render

Every seed is played twice with sides swapped, on the official arena (RULES_VERSION,
OFFICIAL_MAX_STEPS). Same seed => same map and spawns for everyone, so the result is
reproducible.

Agent files run in their OWN PROCESS (see tank_env/remote.py), so both models can live on
one laptop with different venvs (`--python-b path/to/venv/bin/python`). An agent that
crashes, hangs or exceeds `--act-timeout` ms on a single act() forfeits that match.
"""
import argparse
import time

from tank_env import RULES_VERSION, OFFICIAL_MAX_STEPS, TankEnvironment, load_agent, reset_agent
from tank_env.remote import AgentFailure, RemoteAgent

OUTCOMES = {"tank1_win": 1, "tank2_win": 2}


def make_agent(spec, python=None, isolate=True, act_timeout_ms=500.0):
    """Builtin bots run in-process; agent files get their own process unless isolate=False."""
    if isolate and spec.endswith(".py"):
        return RemoteAgent(spec, python=python, timeout_ms=act_timeout_ms)
    return load_agent(spec)


def play_match(agent1, agent2, seed, max_steps=OFFICIAL_MAX_STEPS, env=None, render=False, delay=0.0):
    """
    One match on `seed`. Returns dict(winner=1|2|0, outcome, steps, hp=(h1, h2),
    act_ms=(mean1, max1, mean2, max2), failed=None|1|2, failure=str|None); `winner` is 0 for a draw.
    An agent that raises AgentFailure forfeits (outcome 'forfeit_tankN').
    With render=True it returns None if the user closes the window.
    """
    env = env or TankEnvironment(max_steps=max_steps)
    agents = (agent1, agent2)
    spent, worst = [0.0, 0.0], [0.0, 0.0]
    hp, steps = (100.0, 100.0), 0

    def forfeit(idx, err):
        return {
            "winner": 2 - idx, "outcome": f"forfeit_tank{idx + 1}", "steps": steps, "hp": hp,
            "act_ms": (spent[0] / max(steps, 1), worst[0], spent[1] / max(steps, 1), worst[1]),
            "failed": idx + 1, "failure": str(err),
        }

    for i, agent in enumerate(agents):
        try:
            reset_agent(agent)
        except AgentFailure as e:
            return forfeit(i, e)
    obs1, obs2 = env.reset(seed=seed)
    done, info = False, {}
    while not done:
        actions = []
        for i, (agent, obs) in enumerate(zip(agents, (obs1, obs2))):
            t0 = time.perf_counter()
            try:
                actions.append(agent.act(obs))
            except AgentFailure as e:
                return forfeit(i, e)
            dt = (time.perf_counter() - t0) * 1000.0
            spent[i] += dt
            worst[i] = max(worst[i], dt)
        obs1, obs2, _, done, info = env.step(*actions)
        hp = (obs1["hp"], obs2["hp"])
        steps += 1
        if render:
            if not env.render():
                return None
            if delay > 0:
                time.sleep(delay)
    outcome = info["outcome"]
    return {
        "winner": OUTCOMES.get(outcome, 0),
        "outcome": outcome,
        "steps": steps,
        "hp": hp,
        "act_ms": (spent[0] / steps, worst[0], spent[1] / steps, worst[1]),
        "failed": None,
        "failure": None,
    }


def play_series(agent_a, agent_b, seeds, max_steps=OFFICIAL_MAX_STEPS, render=False, delay=0.0, verbose=False):
    """Plays every seed twice (A on tank 1, then A on tank 2). Returns per-agent totals."""
    stats = {n: dict(wins=0, losses=0, draws=0, forfeits=0, hp=0.0, mean_ms=0.0, max_ms=0.0) for n in "AB"}
    agents = {"A": agent_a, "B": agent_b}
    games = 0
    env = TankEnvironment(max_steps=max_steps)
    for seed in seeds:
        for a_is_tank1 in (True, False):
            who = ("A", "B") if a_is_tank1 else ("B", "A")  # who plays tank1 / tank2
            r = play_match(agents[who[0]], agents[who[1]], seed, max_steps, env, render, delay)
            if r is None:
                env.close()
                return None
            games += 1
            for idx, name in enumerate(who):
                s = stats[name]
                s["hp"] += r["hp"][idx]
                s["mean_ms"] += r["act_ms"][2 * idx]
                s["max_ms"] = max(s["max_ms"], r["act_ms"][2 * idx + 1])
                if r["winner"] == 0:
                    s["draws"] += 1
                elif r["winner"] == idx + 1:
                    s["wins"] += 1
                else:
                    s["losses"] += 1
            if r["failed"]:
                culprit = who[r["failed"] - 1]
                stats[culprit]["forfeits"] += 1
                print(f"  ! {culprit} forfeits seed {seed}: {r['failure'].strip().splitlines()[-1]}")
                if hasattr(agents[culprit], "restart"):
                    agents[culprit].restart()
            if verbose:
                side = "A=tank1" if a_is_tank1 else "A=tank2"
                print(f"  seed {seed:>6} ({side}): {r['outcome']:<13} {r['steps']:>4} steps  "
                      f"HP {r['hp'][0]:5.1f}/{r['hp'][1]:5.1f}")
    for s in stats.values():
        s["hp"] /= games
        s["mean_ms"] /= games
    stats["games"] = games
    env.close()
    return stats


def main():
    p = argparse.ArgumentParser(description="Referee: duel two agents on official rules")
    p.add_argument("agent_a", help="agent .py file, 'rule_based' or 'random'")
    p.add_argument("agent_b", help="agent .py file, 'rule_based' or 'random'")
    p.add_argument("--seeds", type=int, default=20, help="number of maps (each played twice, sides swapped)")
    p.add_argument("--seed-base", type=int, default=1000, help="first map seed (agree on it, keep final seeds secret)")
    p.add_argument("--python-a", help="python interpreter (venv) for agent A, default: this one")
    p.add_argument("--python-b", help="python interpreter (venv) for agent B, default: this one")
    p.add_argument("--act-timeout", type=float, default=500.0,
                   help="ms allowed for one act(); exceeding it forfeits the match (default 500)")
    p.add_argument("--no-isolate", action="store_true", help="run agent files inside the referee process (debugging)")
    p.add_argument("--render", action="store_true", help="show the matches in a window (ESC to stop)")
    p.add_argument("--delay", type=float, default=0.016, help="seconds per tick when rendering")
    p.add_argument("--verbose", "-v", action="store_true", help="print every match")
    args = p.parse_args()

    iso = not args.no_isolate
    a = make_agent(args.agent_a, args.python_a, iso, args.act_timeout)
    b = make_agent(args.agent_b, args.python_b, iso, args.act_timeout)
    seeds = range(args.seed_base, args.seed_base + args.seeds)
    print(f"Rules v{RULES_VERSION} | {args.seeds} maps x 2 sides | max_steps={OFFICIAL_MAX_STEPS}"
          f" | act timeout {args.act_timeout:.0f} ms | {'isolated processes' if iso else 'in-process'}")
    print(f"A = {args.agent_a}\nB = {args.agent_b}\n")

    try:
        res = play_series(a, b, seeds, render=args.render, delay=args.delay, verbose=args.verbose)
    finally:
        for ag in (a, b):
            if hasattr(ag, "close"):
                ag.close()
    if res is None:
        print("Window closed, stopping.")
        return
    print(f"{'':3}{'W':>4}{'L':>4}{'D':>4}{'F':>4}  {'avg HP':>7}  {'act() ms mean/max':>18}")
    for n in "AB":
        s = res[n]
        print(f"{n:<3}{s['wins']:>4}{s['losses']:>4}{s['draws']:>4}{s['forfeits']:>4}  {s['hp']:>7.1f}  "
              f"{s['mean_ms']:>8.3f} / {s['max_ms']:<8.2f}")
    a_w, b_w = res["A"]["wins"], res["B"]["wins"]
    verdict = "A wins" if a_w > b_w else "B wins" if b_w > a_w else "tie"
    print(f"\nResult over {res['games']} games: {verdict} ({a_w}:{b_w})   [F = forfeits, counted as losses]")


if __name__ == "__main__":
    main()
