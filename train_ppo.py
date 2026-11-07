#!/usr/bin/env python3
"""
PPO + GRU + hall-of-fame self-play. Train on a big CPU box / Colab, duel on the laptop.

    PYTHONPATH=../playground/python:. python3 train_ppo.py --save-dir runs/a --workers 11 --envs-per-worker 16
    PYTHONPATH=../playground/python:. python3 train_ppo.py --self-check        # torch vs NumPy policy equivalence

Output (in --save-dir):
    export_latest/{agent.py,policy.npz}   and   export_best/...   -> duel them with referee.py (NumPy only)
    hof/snap_XXXXX.npz                    hall-of-fame snapshots (also usable as agents via NumpyGRUPolicy.load)
    last.pt                               full trainer state; --resume continues after a Colab disconnect

Reward design, opponent mix and what "best" means: see mytrain/selfplay_env.py and TRAINING.md.
"""
import argparse
import collections
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import referee  # shared repo (referee.py)
from tank_env import RuleBasedBot  # shared repo
from mytrain.evalpool import EvalPool
from mytrain.rollout import Rollout
from mytrain.bundle import append_metrics, make_bundle, write_config
from mytrain.gru_policy import FEATURE_DIM, NumpyGRUPolicy, write_standalone_agent
from mytrain.selfplay_env import GAMMA, SelfPlayVecEnv

EVAL_SEED_BASE = 10_000_000  # never used for training maps
LOG2PI = math.log(2.0 * math.pi)


class Net(nn.Module):
    """obs -> tanh(Linear) -> GRU -> actor (3 move means + shoot logit) and critic (value)."""

    def __init__(self, in_dim=FEATURE_DIM, emb=256, hidden=256):
        super().__init__()
        self.hidden = hidden
        self.inp = nn.Linear(in_dim, emb)
        self.gru = nn.GRU(emb, hidden)
        self.actor = nn.Linear(hidden, 4)
        self.critic = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((3,), -0.7))
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, 1.0)
        nn.init.zeros_(self.critic.bias)

    def step(self, x, h, first):
        """x (B,in), h (B,H), first (B,) = 1 where x opens a new episode (memory is wiped)."""
        h = h * (1.0 - first).unsqueeze(1)
        out, _ = self.gru(torch.tanh(self.inp(x)).unsqueeze(0), h.unsqueeze(0))
        h_new = out.squeeze(0)
        return self.actor(h_new), self.critic(h_new).squeeze(1), h_new

    def unroll_reference(self, x, h, first):
        """Plain step-by-step unroll (slow, obviously correct). Kept to test `unroll` against."""
        outs, vals = [], []
        for t in range(x.shape[0]):
            o, v, h = self.step(x[t], h, first[t])
            outs.append(o)
            vals.append(v)
        return torch.stack(outs), torch.stack(vals)

    def unroll(self, x, h, first):
        """
        x (T,B,in), first (T,B) -> raw outputs (T,B,4), values (T,B). Same result as `unroll_reference`,
        but a few fused GRU calls instead of T tiny ones (the update was launch-bound): the window is cut
        only at steps where some env opens a new episode (its memory is wiped there); between cuts the
        whole batch runs as ONE nn.GRU call.
        """
        T = first.shape[0]
        e = torch.tanh(self.inp(x))                                       # (T,B,E)
        cuts = [0] + (first[1:].sum(1) > 0).nonzero().flatten().add(1).tolist() + [T]
        outs = []
        for i in range(len(cuts) - 1):
            s, t = cuts[i], cuts[i + 1]
            h = h * (1.0 - first[s]).unsqueeze(1)                         # wipe memory of envs starting an episode at s
            out, hn = self.gru(e[s:t], h.unsqueeze(0))
            h = hn[0]
            outs.append(out)
        hs = torch.cat(outs)
        return self.actor(hs), self.critic(hs).squeeze(-1)

    def logp_entropy(self, out, move_raw, shoot):
        mu, logits = out[..., :3], out[..., 3]
        std = self.log_std.exp()
        logp_move = (-0.5 * ((move_raw - mu) / std) ** 2 - self.log_std - 0.5 * LOG2PI).sum(-1)
        logp_shoot = -F.binary_cross_entropy_with_logits(logits, shoot, reduction="none")
        ent_move = (0.5 + 0.5 * LOG2PI + self.log_std).sum()
        ent_shoot = F.binary_cross_entropy_with_logits(logits, torch.sigmoid(logits), reduction="none")
        return logp_move + logp_shoot, ent_move + ent_shoot


class CpuEnv:
    """Rust engine in worker processes, wrapped to the same tensor interface as GpuSelfPlayEnv."""

    def __init__(self, vec, device):
        self.vec, self.device, self.eps = vec, device, []

    def reset(self):
        return torch.as_tensor(self.vec.reset(), device=self.device)

    def set_pool(self, pool):
        self.vec.set_pool(pool)

    def step(self, act):
        nxt, rew, done, eps = self.vec.step(act.cpu().numpy())
        self.eps += eps
        t = lambda a: torch.as_tensor(a, device=self.device)
        return t(nxt), t(rew), t(done)

    def pop_stats(self):
        e, self.eps = self.eps, []
        bot = [x for x in e if x["opp"] == "bot"]
        return dict(n=len(e), len=sum(x["steps"] for x in e), win=sum(x["result"] == "win" for x in e),
                    loss=sum(x["result"] == "loss" for x in e), draw=sum(x["result"] == "draw" for x in e),
                    bot_n=len(bot), bot_win=sum(x["result"] == "win" for x in bot))

    def close(self):
        self.vec.close()


def export_params(net: Net) -> dict:
    """torch -> NumpyGRUPolicy layout (value head and log_std stay in the trainer)."""
    sd = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
    return dict(w_in=sd["inp.weight"], b_in=sd["inp.bias"], w_ih=sd["gru.weight_ih_l0"],
                w_hh=sd["gru.weight_hh_l0"], b_ih=sd["gru.bias_ih_l0"], b_hh=sd["gru.bias_hh_l0"],
                w_out=sd["actor.weight"], b_out=sd["actor.bias"], log_std=sd["log_std"])


def write_export(folder: str, params: dict):
    write_standalone_agent(folder, params)


def self_check(emb=32, hidden=32, T=40, B=3) -> float:
    """The exported NumPy policy must reproduce the torch network (including resets). Returns max |diff|."""
    torch.manual_seed(0)
    net = Net(FEATURE_DIM, emb, hidden)
    for p in net.parameters():  # make it non-trivial
        p.data += 0.2 * torch.randn_like(p)
    x = torch.randn(T, B, FEATURE_DIM)
    first = (torch.rand(T, B) < 0.1).float()
    first[0] = 1.0
    with torch.no_grad():
        out, _ = net.unroll(x, torch.zeros(B, hidden), first)
    worst = 0.0
    for b in range(B):
        pol = NumpyGRUPolicy(export_params(net))
        for t in range(T):
            if first[t, b] > 0:
                pol.reset()
            ref = pol.forward(x[t, b].numpy())
            worst = max(worst, float(np.abs(ref - out[t, b].numpy()).max()))
    assert worst < 1e-4, f"numpy policy deviates from torch by {worst}"
    return worst


def score(stats: dict, who: str = "A") -> float:
    s = stats[who]
    return (s["wins"] + 0.5 * s["draws"]) / max(stats["games"], 1)


def evaluate(params: dict, hof: list, maps: int, pool=None):
    """Fixed maps, both sides, on the Rust engine. Returns (score vs rule-based bot, score vs an old snapshot)."""
    seeds = range(EVAL_SEED_BASE, EVAL_SEED_BASE + maps)
    old = hof[1] if len(hof) > 1 else hof[0]
    if pool is not None:
        return pool.scores(params, old, seeds)
    me = NumpyGRUPolicy(params)
    return (score(referee.play_series(me, RuleBasedBot(), seeds)),
            score(referee.play_series(me, NumpyGRUPolicy(old), seeds)))


class _Tee:
    """stdout -> console + <save-dir>/train.log (the bundle carries its head and tail)."""

    def __init__(self, console, file):
        self.console, self.file = console, file

    def write(self, text):
        self.console.write(text)
        self.file.write(text)

    def flush(self):
        self.console.flush()
        self.file.flush()


def autotune_envs(bank, device, hidden, slots, max_envs, floor=1024, gain=1.15):
    """
    Benchmark the GPU arena at doubling sizes. The step is launch-bound, so throughput grows almost linearly
    until the GPU saturates; stop at the first size that is not at least `gain` x faster than the previous one
    (or at an out-of-memory error / `max_envs`). Returns ([(envs, steps/s), ...], chosen_envs).
    """
    from mytrain.gpu_selfplay import GpuSelfPlayEnv
    from mytrain.gru_policy import random_params

    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    pool = [random_params(FEATURE_DIM, hidden, hidden, 0)]
    results, e = [], floor
    while e <= max_envs:
        try:
            env = GpuSelfPlayEnv(e, bank, device, hidden, slots, 0.2)
            env.set_pool(pool)
            env.reset()
            act = torch.zeros(e, 4, device=device)
            for _ in range(3):
                env.step(act)
            sync()
            t0 = time.time()
            for _ in range(6):
                env.step(act)
            sync()
            sps = 6 * e / (time.time() - t0)
        except RuntimeError as err:  # CUDA out of memory
            if "out of memory" not in str(err).lower():
                raise
            break
        finally:
            env = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results.append((e, sps))
        if len(results) > 1 and sps < results[-2][1] * gain:
            break
        e *= 2
    chosen = results[-2][0] if len(results) > 1 and results[-1][1] < results[-2][1] * gain else results[-1][0]
    return results, chosen


def main():
    ap = argparse.ArgumentParser(description="PPO + GRU + hall-of-fame self-play")
    ap.add_argument("--save-dir", default="runs/ppo")
    ap.add_argument("--steps", type=float, default=50e6, help="total environment steps (learner side)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--env", choices=["auto", "cpu", "gpu"], default="auto",
                    help="where the arenas run: gpu = tensor re-implementation of the arena (thousands of envs, no CPU "
                         "bottleneck); cpu = the Rust engine in worker processes; auto = gpu if CUDA is available")
    ap.add_argument("--gpu-envs", type=int, default=0,
                    help="parallel arenas for --env gpu; 0 = auto: benchmark where the GPU saturates, but keep at least "
                         "--min-updates PPO updates in the --steps budget (more arenas than that makes learning worse, not faster)")
    ap.add_argument("--max-envs", type=int, default=65536, help="upper bound for the auto benchmark")
    ap.add_argument("--min-updates", type=int, default=400, help="auto sizing keeps at least this many PPO updates per run")
    ap.add_argument("--slots", type=int, default=6, help="hall-of-fame snapshots on stage at once (--env gpu)")
    ap.add_argument("--map-bank", type=int, default=20000, help="Rust-generated (official) maps for --env gpu")
    ap.add_argument("--map-mix", type=float, default=0.3,
                    help="share of extra TRAINING-ONLY arenas (open, sparse, dense, close, free spawns) next to the official maps, "
                         "so movement does not depend on one layout; 0 = official maps only. Evaluation is always official.")
    ap.add_argument("--target-envs", type=int, default=176,
                    help="total arenas; the same PPO batch on any machine (spread over the workers)")
    ap.add_argument("--envs-per-worker", type=int, default=0, help="0 = auto: ceil(target-envs / workers)")
    ap.add_argument("--rollout", type=int, default=128, help="steps per env per update")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=0, help="env chunks per epoch (sequences stay whole); 0 = auto: 4 (CPU env) / 8 (GPU env)")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--reward-scale", type=float, default=0.05)
    ap.add_argument("--bot-prob", type=float, default=0.2, help="share of episodes against the rule-based anchor")
    ap.add_argument("--latest-prob", type=float, default=0.35, help="share of hall-of-fame games against the 2 newest snapshots")
    ap.add_argument("--snapshot-every", type=int, default=0, help="updates between hall-of-fame snapshots; 0 = auto (10 CPU env / 2 GPU env)")
    ap.add_argument("--hof-size", type=int, default=60, help="hall-of-fame snapshots kept (old ones are thinned out at random)")
    ap.add_argument("--lr-final-frac", type=float, default=0.1, help="learning rate decays linearly to this fraction of --lr by the end of --steps")
    ap.add_argument("--eval-every", type=int, default=0, help="updates between evaluations on fixed maps; 0 = auto (20 CPU env / 8 GPU env)")
    ap.add_argument("--cuda-graph", choices=["auto", "on", "off"], default="auto",
                    help="record the whole rollout step as a CUDA graph (replayed with one launch: removes the CPU launch bottleneck "
                         "of the GPU arena). auto = on for --env gpu on CUDA; it benchmarks itself and falls back to eager on any problem")
    ap.add_argument("--eval-procs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="CPU processes for the Rust-engine evaluation (1 = in the trainer process)")
    ap.add_argument("--eval-maps", type=int, default=12, help="fixed maps per evaluation (x2 sides); few maps make the *best* pick a coin flip")
    ap.add_argument("--save-every", type=int, default=0, help="updates between full checkpoints (last.pt); 0 = auto (10 CPU env / 4 GPU env)")
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--updates", type=int, default=None, help="override the update count (smoke tests)")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        print(f"self-check OK: NumPy policy matches torch (max |diff| = {self_check():.2e})")
        return

    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(args.save_dir, "hof"), exist_ok=True)
    log_file = open(os.path.join(args.save_dir, "train.log"), "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, log_file)
    print(f"=== session start {time.strftime('%Y-%m-%d %H:%M:%S')} | torch {torch.__version__} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'} | {os.cpu_count()} vCPU ===", flush=True)
    net = Net(FEATURE_DIM, args.hidden, args.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    use_gpu = args.env == "gpu" or (args.env == "auto" and device.type == "cuda")
    args.snapshot_every = args.snapshot_every or (2 if use_gpu else 10)
    args.eval_every = args.eval_every or (8 if use_gpu else 20)
    args.save_every = args.save_every or (4 if use_gpu else 10)
    args.envs_per_worker = args.envs_per_worker or max(1, math.ceil(args.target_envs / args.workers))
    T = args.rollout
    bank = None
    if use_gpu:
        from mytrain.gpu_arena import MapBank

        seeds = random.Random(args.seed).sample(range(5_000_000), args.map_bank)  # training maps: seeds < 5M
        print(f"building {args.map_bank} Rust maps for the GPU arena ...", flush=True)
        extra = []
        if args.map_mix > 0:
            from mytrain.mapmix import build_mix

            extra = build_mix(int(args.map_bank * args.map_mix / (1 - args.map_mix)), args.seed)
            print(f"  + {len(extra)} training-only arenas (open/sparse/dense/close/free)", flush=True)
        bank = MapBank.from_rust(seeds, device, extra)
        if args.gpu_envs:
            N = args.gpu_envs
        else:
            if device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True  # A100: ~8x faster matmuls for the opponents
                torch.backends.cudnn.allow_tf32 = True
                table, saturated = autotune_envs(bank, device, args.hidden, args.slots, args.max_envs)
            else:  # a CPU cannot "saturate": keep a modest fixed size
                table, saturated = [], 512
            by_budget = max(1024, 2 ** int(math.log2(max(1.0, args.steps / (T * args.min_updates)))))
            N = min(saturated, by_budget)
            print("auto-sizing the GPU arena: " + ", ".join(f"{e}: {sps / 1e3:,.0f}k sps" for e, sps in table))
            print(f"  GPU saturates at {saturated} envs; the {args.steps / 1e6:,.0f}M-step budget allows {by_budget} "
                  f"(>= {args.min_updates} updates) -> using {N}. More GPU use = a bigger --steps budget.")
    else:
        N = args.workers * args.envs_per_worker
    args.minibatches = args.minibatches or (max(8, math.ceil(N / 2048)) if use_gpu else 4)
    total_updates = args.updates or max(1, int(args.steps // (N * T)))
    update0, steps_done, best_score = 0, 0, -1.0
    hof = [export_params(net)]  # update 0: an untrained snapshot, so the very first games are easy
    pinned = []                 # every new *best* export stays an opponent for good (guards against forgetting / cycling)
    last_path = os.path.join(args.save_dir, "last.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        net.load_state_dict(ck["net"]), opt.load_state_dict(ck["opt"])
        update0, steps_done, best_score, hof = ck["update"], ck["steps"], ck["best"], ck["hof"]
        pinned = ck.get("pinned", [])
        print(f"resumed from update {update0} ({steps_done:,} steps, hof={len(hof)})")
        if not args.updates:  # the batch size may differ from the run that wrote the checkpoint: count what is left in steps
            total_updates = update0 + max(0, math.ceil((args.steps - steps_done) / (N * T)))

    layout = "GPU arena" if use_gpu else f"Rust x {args.workers}x{args.envs_per_worker}"
    print(f"device={device} envs={N} ({layout}) rollout={T} "
          f"batch={N * T:,}/update updates={total_updates} params={sum(p.numel() for p in net.parameters()):,}")
    write_config(args.save_dir, args, dict(device=str(device), num_envs=N, total_updates=total_updates,
                                           params=sum(p.numel() for p in net.parameters())))
    if use_gpu:
        from mytrain.gpu_selfplay import GpuSelfPlayEnv

        env = GpuSelfPlayEnv(N, bank, device, args.hidden, args.slots, args.bot_prob)
    else:
        env = CpuEnv(SelfPlayVecEnv(args.workers, args.envs_per_worker, args.seed, args.bot_prob, args.latest_prob), device)
    env.set_pool(pinned + hof)

    want_graph = use_gpu and device.type == "cuda" and args.cuda_graph != "off"
    ro = Rollout(net, env, T, args.hidden, device, args.reward_scale, FEATURE_DIM, graph=want_graph)
    print(f"rollout: {ro.graph_note}", flush=True)
    eval_pool = None
    window = collections.deque(maxlen=4)  # episode stats of the last updates

    try:
        for update in range(update0 + 1, total_updates + 1):
            t0 = time.time()
            lr = args.lr * (1.0 - (1.0 - args.lr_final_frac) * min(1.0, steps_done / args.steps))
            for group in opt.param_groups:
                group["lr"] = lr
            h0 = ro.h.clone()
            # ---------------- rollout (T steps; eager or one CUDA-graph launch per step) ----------------
            ro.run()
            B_obs, B_first, B_move, B_shoot, B_logp, B_val, B_rew, B_done = (
                ro.obs_buf, ro.first_buf, ro.move_buf, ro.shoot_buf, ro.logp_buf, ro.val_buf, ro.rew_buf, ro.done_buf)
            window.append(env.pop_stats())
            steps_done += N * T
            t_roll = time.time() - t0
            # ---------------- GAE ----------------
            with torch.no_grad():
                _, last_val, _ = net.step(ro.obs, ro.h, ro.first)
            adv = torch.zeros(T, N, device=device)
            gae = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                next_val = last_val if t == T - 1 else B_val[t + 1]
                live = 1.0 - B_done[t]
                delta = B_rew[t] + GAMMA * next_val * live - B_val[t]
                gae = delta + GAMMA * args.lam * live * gae
                adv[t] = gae
            ret = adv + B_val
            # ---------------- PPO update ----------------
            stats = collections.defaultdict(list)
            for _ in range(args.epochs):
                for idx in torch.randperm(N, device=device).chunk(args.minibatches):
                    out, val = net.unroll(B_obs[:, idx], h0[idx], B_first[:, idx])
                    logp, ent = net.logp_entropy(out, B_move[:, idx], B_shoot[:, idx])
                    log_ratio = logp - B_logp[:, idx]
                    ratio = log_ratio.exp()
                    a = adv[:, idx]
                    a = (a - a.mean()) / (a.std() + 1e-8)
                    pg = torch.max(-a * ratio, -a * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                    v_loss = 0.5 * ((val - ret[:, idx]) ** 2).mean()
                    loss = pg + args.vf_coef * v_loss - args.ent_coef * ent.mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    opt.step()
                    stats["pg"].append(pg.item()), stats["v"].append(v_loss.item())
                    stats["ent"].append(ent.mean().item())
                    stats["kl"].append(((ratio - 1) - log_ratio).mean().item())
            # ---------------- hall of fame / eval / save ----------------
            if update % args.snapshot_every == 0:
                hof.append(export_params(net))
                if len(hof) > args.hof_size:
                    del hof[random.randrange(0, len(hof) - 4)]  # thin out old ones, keep the newest 4
                np.savez_compressed(os.path.join(args.save_dir, "hof", f"snap_{update:05d}.npz"), **hof[-1])
                env.set_pool(pinned + hof)
            line, ev_bot, ev_old = "", "", ""
            if update % args.eval_every == 0 or update == total_updates:
                params = export_params(net)
                if eval_pool is None and args.eval_procs > 1:
                    eval_pool = EvalPool(args.eval_procs)  # first evaluation: start the worker processes once
                vs_bot, vs_old = evaluate(params, hof, args.eval_maps, eval_pool)
                ev_bot, ev_old = vs_bot, vs_old
                line = f" | EVAL vs_bot {vs_bot:.2f} vs_old {vs_old:.2f}"
                write_export(os.path.join(args.save_dir, "export_latest"), params)
                if vs_bot + vs_old > best_score:
                    best_score = vs_bot + vs_old
                    write_export(os.path.join(args.save_dir, "export_best"), params)
                    line += " *best"
                    pinned.append(params)
                    del pinned[:-12]  # at most 12 pinned bests
                    env.set_pool(pinned + hof)
            if update % args.save_every == 0 or update == total_updates:
                torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), update=update, steps=steps_done,
                                best=best_score, hof=hof, pinned=pinned), last_path)

            tot = lambda key: sum(w[key] for w in window)

            def rate(kind, res):
                n, k = (tot("n"), tot(res)) if kind == "all" else (tot("bot_n"), tot("bot_win"))
                return k / n if n else float("nan")

            ep_len = tot("len") / tot("n") if tot("n") else 0.0

            dt = time.time() - t0
            vram = f" vram {torch.cuda.max_memory_allocated() / 1e9:.1f}GB" if device.type == "cuda" else ""
            print(f"upd {update:5d}/{total_updates} | {steps_done / 1e6:6.2f}M steps | {N * T / dt:7.0f} sps | "
                  f"win {rate('all', 'win'):.2f} loss {rate('all', 'loss'):.2f} draw {rate('all', 'draw'):.2f} "
                  f"(bot win {rate('bot', 'win'):.2f}) | ep_len {ep_len:5.0f} "
                  f"| pg {np.mean(stats['pg']):+.3f} v {np.mean(stats['v']):.3f} ent {np.mean(stats['ent']):.2f} "
                  f"kl {np.mean(stats['kl']):.4f} | roll {t_roll:.1f}s upd {dt - t_roll:.1f}s{vram}{line}", flush=True)
            append_metrics(args.save_dir, dict(
                update=update, steps=steps_done, sps=round(N * T / dt), win=rate('all', 'win'), loss=rate('all', 'loss'),
                draw=rate('all', 'draw'), bot_win=rate('bot', 'win'),
                ep_len=ep_len if tot('n') else "",
                pg=np.mean(stats['pg']), v=np.mean(stats['v']), ent=np.mean(stats['ent']), kl=np.mean(stats['kl']),
                eval_vs_bot=ev_bot, eval_vs_old=ev_old, time=round(time.time())))
            if ev_bot != "":
                make_bundle(args.save_dir)  # always-fresh hand-off folder on Drive
    finally:
        env.close()
        if eval_pool is not None:
            eval_pool.close()

    write_export(os.path.join(args.save_dir, "export_latest"), export_params(net))
    make_bundle(args.save_dir)
    print(f"bundle for review: {os.path.join(args.save_dir, 'bundle')}")
    print(f"done. duel it: PYTHONPATH=../playground/python python3 ../playground/python/referee.py {args.save_dir}/export_best/agent.py rule_based")


if __name__ == "__main__":
    main()
