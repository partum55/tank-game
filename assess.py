#!/usr/bin/env python3
"""
Verdict for a training bundle (folder or .zip made by the trainer): is the agent trained, or does it need more?

    PYTHONPATH=../playground/python:. python3 assess.py path/to/bundle.zip [--maps 30]

Looks at (1) the learning curve in metrics.csv, (2) fresh duels on maps the trainer never used:
latest/best vs the rule-based bot and latest vs older hall-of-fame snapshots. "Still improving" means the
newest model reliably beats the one from earlier in the same run - then more training should help.
The last word is always the referee against the real opponent.
"""
import argparse
import csv
import glob
import json
import os
import tempfile
import zipfile

import numpy as np

import referee
import tank_env
from mytrain.gru_policy import NumpyGRUPolicy
from tank_env import RuleBasedBot

ASSESS_SEED_BASE = 20_000_000  # trainer evaluates on 10_000_000.., trains on < 5_000_000


def open_bundle(path: str) -> str:
    if os.path.isdir(path):
        return path
    out = tempfile.mkdtemp(prefix="bundle_")
    with zipfile.ZipFile(path) as z:
        z.extractall(out)
    inner = [d for d in glob.glob(os.path.join(out, "*")) if os.path.isdir(d)]
    return out if os.path.exists(os.path.join(out, "metrics.csv")) or not inner else inner[0]


def read_metrics(folder: str) -> list:
    rows = []
    with open(os.path.join(folder, "metrics.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ("", None) else None) for k, v in r.items()})
    return rows


def window(rows, key, lo, hi):
    n = len(rows)
    vals = [r[key] for r in rows[int(n * lo):max(int(n * hi), int(n * lo) + 1)]
            if r.get(key) is not None and r[key] == r[key]]  # skip empty / NaN
    return float(np.mean(vals)) if vals else float("nan")


def duel(a, b, maps: int):
    res = referee.play_series(a, b, range(ASSESS_SEED_BASE, ASSESS_SEED_BASE + maps))
    s = res["A"]
    return (s["wins"] + 0.5 * s["draws"]) / res["games"], s["wins"], s["losses"], s["draws"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--maps", type=int, default=30, help="maps per duel (each played on both sides)")
    args = ap.parse_args()
    root = open_bundle(args.bundle)

    cfg = json.load(open(os.path.join(root, "config.json")))
    rows = read_metrics(root)
    last = rows[-1]
    print(f"== run: {int(last['steps'] / 1e6)}M steps, {int(last['update'])} updates | rules {cfg.get('rules_version')} "
          f"(here {tank_env.RULES_VERSION}) | arena {cfg.get('arena_git')} trainer {cfg.get('trainer_git')} | "
          f"envs {cfg.get('num_envs')} hidden {cfg.get('hidden')} | {len(cfg.get('runs', []))} session(s)")
    if cfg.get("rules_version") != tank_env.RULES_VERSION:
        print("!! rules version differs from the arena checked out here - results are not comparable")

    log = os.path.join(root, "train_log.txt")
    if os.path.exists(log):
        keys = ("session start", "device=", "auto-sizing", "saturates", "resumed")
        print("\n== setup (from the training log)")
        for line in open(log, errors="replace"):
            if any(k in line for k in keys):
                print("  " + line.rstrip()[:200])
        tail = [l for l in open(log, errors="replace").read().splitlines() if l.startswith("upd")][-1:]
        print("  last:", tail[0][:200] if tail else "-")

    print("\n== curve (win/loss/draw: last 400 episodes vs hall of fame + bot)")
    print(f"{'steps':>7} {'win':>5} {'draw':>5} {'botwin':>6} {'ep_len':>6} {'ent':>5}  eval(vs_bot/vs_old)")
    idx = sorted({int(i * (len(rows) - 1) / 9) for i in range(10)})
    for i in idx:
        r = rows[i]
        ev = [x for x in rows[max(0, i - 24):i + 1] if x["eval_vs_bot"] is not None]
        e = f"{ev[-1]['eval_vs_bot']:.2f}/{ev[-1]['eval_vs_old']:.2f}" if ev else "-"
        f = lambda k, p=2: "-" if r[k] is None else f"{r[k]:.{p}f}"
        print(f"{r['steps'] / 1e6:6.1f}M {f('win'):>5} {f('draw'):>5} {f('bot_win'):>6} {f('ep_len', 0):>6} {f('ent'):>5}  {e}")

    print("\n== trend: last 20% of the run vs the 20% before")
    ev_rows = [r for r in rows if r["eval_vs_old"] is not None]
    trend = {}
    for key, src in (("win", rows), ("draw", rows), ("bot_win", rows), ("eval_vs_bot", ev_rows), ("eval_vs_old", ev_rows)):
        new, old = window(src, key, 0.8, 1.0), window(src, key, 0.6, 0.8)
        trend[key] = new - old
        print(f"  {key:12s} {old:5.2f} -> {new:5.2f}  ({new - old:+.2f})")

    agents = {}
    for name in ("export_latest", "export_best"):
        p = os.path.join(root, name, "policy.npz")
        if os.path.exists(p):
            agents[name] = NumpyGRUPolicy.load(p)
    snaps = sorted(glob.glob(os.path.join(root, "hof", "snap_*.npz")))
    latest = agents.get("export_latest") or agents["export_best"]

    print(f"\n== fresh duels on unseen maps ({args.maps} maps x 2 sides = {args.maps * 2} games each; score = (W + D/2) / games)")
    bot = RuleBasedBot()
    scores = {}
    for name, ag in agents.items():
        sc, w, l, d = duel(ag, bot, args.maps)
        scores[name + "_vs_bot"] = sc
        print(f"  {name:13s} vs rule_based : {sc:.2f}   (W{w} L{l} D{d})")
    snap_scores = []
    if snaps:
        pick = sorted({snaps[round(i * (len(snaps) - 1) / 3)] for i in range(4)})  # oldest .. newest
        for s in pick:
            sc, w, l, d = duel(latest, NumpyGRUPolicy.load(s), args.maps)
            snap_scores.append((os.path.basename(s), sc))
            print(f"  latest        vs {os.path.basename(s):15s}: {sc:.2f}   (W{w} L{l} D{d})")

    vs_bot = scores.get("export_latest_vs_bot", scores.get("export_best_vs_bot", 0.0))
    mid = snap_scores[len(snap_scores) // 2][1] if snap_scores else float("nan")
    old_end = snap_scores[0][1] if snap_scores else float("nan")
    improving = (mid >= 0.60) or (trend["eval_vs_old"] > 0.05 and trend["win"] > 0.0)
    strong = vs_bot >= 0.85
    print("\n== verdict")
    if last["steps"] < 5e6 or last["update"] < 250:
        print(f"  TOO EARLY to judge ({last['steps'] / 1e6:.1f}M steps, {int(last['update'])} updates; need >= 5M steps AND >= 250 updates): "
              "PPO learns per update, so a run with few huge updates has barely started. Keep training.")
    print(f"  vs bot {vs_bot:.2f} | vs mid-run snapshot {mid:.2f} | vs oldest snapshot {old_end:.2f} | "
          f"eval-vs-old trend {trend['eval_vs_old']:+.2f}")
    if improving and not strong:
        print("  NOT DONE YET: still clearly improving against its own past and not dominant vs the bot -> continue (--resume, higher --steps).")
    elif improving and strong:
        print("  STRONG and still improving: usable now; more steps will still add strength.")
    elif strong:
        print("  PLATEAU, strong vs the bot: more of the same will add little. Time to duel the real opponent; "
              "change something (reward weights, bot-prob, hall-of-fame mix) only if it loses there.")
    else:
        print("  STUCK: not improving and not strong vs the bot -> do not just run longer; check draw rate / entropy "
              "(raise --ent-coef, shaping weights, lower --bot-prob) or the learning rate.")
    print("  (self-play scores are relative to this run's own history; the referee vs the real opponent decides.)")


if __name__ == "__main__":
    main()
