"""
Training telemetry + the hand-off bundle.

Every update the trainer appends a row to metrics.csv; at every evaluation (and at the end) it refreshes
`<save-dir>/bundle/`: one small folder (~15 MB) with everything needed to judge a run from outside:
    config.json        settings, rules version, git revisions
    metrics.csv        per-update curve (win/loss/draw, vs rule-based, eval scores, losses, speed)
    export_best/ export_latest/    self-contained agents (agent.py + policy.npz + tank_gru_policy.py)
    train_log.txt      head (device, GPU sizing) + tail of the console output
    hof/snap_*.npz     up to 6 evenly spaced hall-of-fame snapshots (oldest ... newest)
Zip it and hand it over; `python assess.py <bundle-or-zip>` turns it into a verdict.
"""
import csv
import glob
import json
import os
import shutil
import subprocess
import time

METRIC_FIELDS = ["update", "steps", "sps", "win", "loss", "draw", "bot_win", "ep_len",
                 "pg", "v", "ent", "kl", "eval_vs_bot", "eval_vs_old", "time"]


def _git_rev(path: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def write_config(save_dir: str, args, extra: dict):
    import tank_env

    cfg = dict(vars(args), **extra)
    cfg["rules_version"] = tank_env.RULES_VERSION
    cfg["arena_git"] = _git_rev(os.path.dirname(os.path.dirname(tank_env.__file__)))
    cfg["trainer_git"] = _git_rev(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(save_dir, "config.json")
    runs = []
    if os.path.exists(path):  # resumed run: keep the history of settings
        prev = json.load(open(path))
        runs = prev.get("runs", [prev])
    runs.append(dict(cfg, started=time.strftime("%Y-%m-%d %H:%M:%S")))
    json.dump(dict(cfg, runs=runs), open(path, "w"), indent=1, default=str)


def append_metrics(save_dir: str, row: dict):
    path = os.path.join(save_dir, "metrics.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, METRIC_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in METRIC_FIELDS})


def make_bundle(save_dir: str, max_snaps: int = 6) -> str:
    out = os.path.join(save_dir, "bundle")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    for name in ("config.json", "metrics.csv"):
        if os.path.exists(os.path.join(save_dir, name)):
            shutil.copy(os.path.join(save_dir, name), out)
    for name in ("export_best", "export_latest"):
        if os.path.isdir(os.path.join(save_dir, name)):
            shutil.copytree(os.path.join(save_dir, name), os.path.join(out, name))
    log = os.path.join(save_dir, "train.log")
    if os.path.exists(log):  # setup (device, GPU sizing table) at the top, recent progress at the bottom
        lines = open(log, errors="replace").read().splitlines()
        if len(lines) > 400:
            lines = lines[:80] + [f"... ({len(lines) - 380} lines trimmed) ..."] + lines[-300:]
        open(os.path.join(out, "train_log.txt"), "w").write("\n".join(lines) + "\n")
    snaps = sorted(glob.glob(os.path.join(save_dir, "hof", "snap_*.npz")))
    if snaps:
        k = min(max_snaps, len(snaps))
        pick = sorted({snaps[round(i * (len(snaps) - 1) / max(k - 1, 1))] for i in range(k)})
        os.makedirs(os.path.join(out, "hof"))
        for s in pick:
            shutil.copy(s, os.path.join(out, "hof"))
    return out
