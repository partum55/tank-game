#!/usr/bin/env python3
"""
Pick the strongest agent of a bundle by a round-robin tournament on the Rust engine (unseen maps, both sides, in parallel).

    PYTHONPATH=../playground/python:. python3 pick_best.py bundle.zip [--maps 8] [--extra path/to/policy.npz ...] [--out runs/final]

Candidates: export_best, export_latest, every hof/snap_*.npz in the bundle, the rule-based bot, plus --extra policies.
Ranking = mean score against all the others (score = (W + D/2) / games). The winner is written as a ready-to-duel agent folder
(agent.py + policy.npz + tank_gru_policy.py). Self-play strength is relative: the final judge is still the real opponent.
"""
import argparse
import glob
import itertools
import multiprocessing as mp
import os

import numpy as np

from assess import open_bundle

SEED_BASE = 60_000_000


def _load(spec):
    from tank_env import RuleBasedBot
    from mytrain.gru_policy import NumpyGRUPolicy

    return RuleBasedBot() if spec == "bot" else NumpyGRUPolicy.load(spec)


def _match(job):
    a, b, seeds = job
    import referee

    res = referee.play_series(_load(a), _load(b), seeds)
    s = res["A"]
    return a, b, (s["wins"] + 0.5 * s["draws"]) / res["games"], s["wins"], s["losses"], s["draws"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--maps", type=int, default=8, help="maps per pairing (each played on both sides)")
    ap.add_argument("--extra", nargs="*", default=[], help="more policy.npz files to include")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=None, help="write the winner here as an agent folder")
    args = ap.parse_args()

    root = open_bundle(args.bundle)
    cands = {"bot": "bot"}
    for name in ("export_best", "export_latest"):
        p = os.path.join(root, name, "policy.npz")
        if os.path.exists(p):
            cands[name] = p
    for p in sorted(glob.glob(os.path.join(root, "hof", "snap_*.npz"))):
        cands[os.path.basename(p)[:-4]] = p
    for p in args.extra:
        cands[os.path.basename(os.path.dirname(os.path.abspath(p))) or p] = p
    names = list(cands)
    seeds = list(range(SEED_BASE, SEED_BASE + args.maps))
    jobs = [(cands[a], cands[b], seeds) for a, b in itertools.combinations(names, 2)]
    print(f"{len(names)} candidates, {len(jobs)} pairings x {args.maps * 2} games on {args.procs} processes ...", flush=True)
    path_to_name = {v: k for k, v in cands.items()}
    with mp.get_context("spawn").Pool(args.procs) as pool:
        results = pool.map(_match, jobs, chunksize=1)

    M = {}
    for a, b, sc, *_ in results:
        M[path_to_name[a], path_to_name[b]] = sc
        M[path_to_name[b], path_to_name[a]] = 1.0 - sc
    mean = {n: float(np.mean([M[n, m] for m in names if m != n])) for n in names}
    order = sorted(names, key=lambda n: -mean[n])
    noisy = {n: n == "bot" or "log_std" in np.load(cands[n]).files for n in names}
    print(f"\n{'rank':>4} {'agent':22s} {'mean':>5} {'vs bot':>7}")
    for i, n in enumerate(order, 1):
        note = "" if noisy[n] else "   (no log_std in the file: plays deterministically - an older export, probably weaker than it is)"
        print(f"{i:>4} {n:22s} {mean[n]:5.2f} {('-' if n == 'bot' else format(M[n, 'bot'], '.2f')):>7}{note}")
    winner = next(n for n in order if n != "bot")
    print(f"\nwinner: {winner} (mean {mean[winner]:.2f}, vs bot {M[winner, 'bot']:.2f})")
    if args.out:
        from mytrain.gru_policy import OPTIONAL_KEYS, PARAM_KEYS, write_standalone_agent

        d = np.load(cands[winner])
        write_standalone_agent(args.out, {k: d[k] for k in PARAM_KEYS + OPTIONAL_KEYS if k in d.files})
        print(f"written: {args.out}/agent.py  (noise stored in the weights: {'log_std' in d.files})")


if __name__ == "__main__":
    main()
