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
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import referee  # shared repo (referee.py)
from tank_env import RuleBasedBot  # shared repo
from mytrain.bundle import append_metrics, make_bundle, write_config
from mytrain.gru_policy import FEATURE_DIM, NumpyGRUPolicy, write_standalone_agent
from mytrain.selfplay_env import GAMMA, SelfPlayVecEnv

EVAL_SEED_BASE = 10_000_000  # never used for training maps
LOG2PI = math.log(2.0 * math.pi)


class Net(nn.Module):
    """obs -> tanh(Linear) -> GRU -> actor (3 move means + shoot logit) and critic (value)."""

    def __init__(self, in_dim=27, emb=256, hidden=256):
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


def export_params(net: Net) -> dict:
    """torch -> NumpyGRUPolicy layout (value head and log_std stay in the trainer)."""
    sd = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
    return dict(w_in=sd["inp.weight"], b_in=sd["inp.bias"], w_ih=sd["gru.weight_ih_l0"],
                w_hh=sd["gru.weight_hh_l0"], b_ih=sd["gru.bias_ih_l0"], b_hh=sd["gru.bias_hh_l0"],
                w_out=sd["actor.weight"], b_out=sd["actor.bias"])


def write_export(folder: str, params: dict):
    write_standalone_agent(folder, params)


def self_check(emb=32, hidden=32, T=40, B=3) -> float:
    """The exported NumPy policy must reproduce the torch network (including resets). Returns max |diff|."""
    torch.manual_seed(0)
    net = Net(27, emb, hidden)
    for p in net.parameters():  # make it non-trivial
        p.data += 0.2 * torch.randn_like(p)
    x = torch.randn(T, B, 27)
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


def evaluate(params: dict, hof: list, maps: int):
    """Fixed maps, both sides. Returns (score vs rule-based bot, score vs an old snapshot)."""
    seeds = range(EVAL_SEED_BASE, EVAL_SEED_BASE + maps)
    me = NumpyGRUPolicy(params)
    vs_bot = score(referee.play_series(me, RuleBasedBot(), seeds))
    old = NumpyGRUPolicy(hof[1] if len(hof) > 1 else hof[0])
    vs_old = score(referee.play_series(me, old, seeds))
    return vs_bot, vs_old


def main():
    ap = argparse.ArgumentParser(description="PPO + GRU + hall-of-fame self-play")
    ap.add_argument("--save-dir", default="runs/ppo")
    ap.add_argument("--steps", type=float, default=50e6, help="total environment steps (learner side)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--target-envs", type=int, default=176,
                    help="total arenas; the same PPO batch on any machine (spread over the workers)")
    ap.add_argument("--envs-per-worker", type=int, default=0, help="0 = auto: ceil(target-envs / workers)")
    ap.add_argument("--rollout", type=int, default=128, help="steps per env per update")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4, help="env chunks per epoch (sequences stay whole)")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--reward-scale", type=float, default=0.05)
    ap.add_argument("--bot-prob", type=float, default=0.2, help="share of episodes against the rule-based anchor")
    ap.add_argument("--latest-prob", type=float, default=0.35, help="share of hall-of-fame games against the 2 newest snapshots")
    ap.add_argument("--snapshot-every", type=int, default=10, help="updates between hall-of-fame snapshots")
    ap.add_argument("--hof-size", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-maps", type=int, default=6)
    ap.add_argument("--save-every", type=int, default=10)
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
    net = Net(FEATURE_DIM, args.hidden, args.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    args.envs_per_worker = args.envs_per_worker or max(1, math.ceil(args.target_envs / args.workers))
    N, T = args.workers * args.envs_per_worker, args.rollout
    total_updates = args.updates or max(1, int(args.steps // (N * T)))
    update0, steps_done, best_score = 0, 0, -1.0
    hof = [export_params(net)]  # update 0: an untrained snapshot, so the very first games are easy
    last_path = os.path.join(args.save_dir, "last.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        net.load_state_dict(ck["net"]), opt.load_state_dict(ck["opt"])
        update0, steps_done, best_score, hof = ck["update"], ck["steps"], ck["best"], ck["hof"]
        print(f"resumed from update {update0} ({steps_done:,} steps, hof={len(hof)})")

    print(f"device={device} envs={N} ({args.workers}x{args.envs_per_worker}) rollout={T} "
          f"batch={N * T:,}/update updates={total_updates} params={sum(p.numel() for p in net.parameters()):,}")
    write_config(args.save_dir, args, dict(device=str(device), num_envs=N, total_updates=total_updates,
                                           params=sum(p.numel() for p in net.parameters())))
    env = SelfPlayVecEnv(args.workers, args.envs_per_worker, args.seed, args.bot_prob, args.latest_prob)
    env.set_pool(hof)

    obs = torch.as_tensor(env.reset(), device=device)
    h = torch.zeros(N, args.hidden, device=device)
    first = torch.ones(N, device=device)
    recent = collections.deque(maxlen=400)
    B_obs = torch.zeros(T, N, FEATURE_DIM, device=device)
    B_first, B_shoot, B_logp, B_val, B_rew, B_done = (torch.zeros(T, N, device=device) for _ in range(6))
    B_move = torch.zeros(T, N, 3, device=device)

    try:
        for update in range(update0 + 1, total_updates + 1):
            t0 = time.time()
            h0 = h.clone()
            # ---------------- rollout ----------------
            for t in range(T):
                B_obs[t], B_first[t] = obs, first
                with torch.no_grad():
                    out, val, h = net.step(obs, h, first)
                    std = net.log_std.exp()
                    move = out[:, :3] + std * torch.randn_like(out[:, :3])
                    shoot = torch.bernoulli(torch.sigmoid(out[:, 3]))
                    logp, _ = net.logp_entropy(out, move, shoot)
                act = torch.cat([move.clamp(-1, 1), (shoot * 2 - 1).unsqueeze(1)], dim=1).cpu().numpy()
                nxt, rew, done, episodes = env.step(act)
                B_move[t], B_shoot[t], B_logp[t], B_val[t] = move, shoot, logp, val
                B_rew[t] = torch.as_tensor(rew, device=device) * args.reward_scale
                B_done[t] = torch.as_tensor(done, device=device)
                obs, first = torch.as_tensor(nxt, device=device), B_done[t]
                recent.extend(episodes)
            steps_done += N * T
            t_roll = time.time() - t0
            # ---------------- GAE ----------------
            with torch.no_grad():
                _, last_val, _ = net.step(obs, h, first)
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
                env.set_pool(hof)
            line, ev_bot, ev_old = "", "", ""
            if update % args.eval_every == 0 or update == total_updates:
                params = export_params(net)
                vs_bot, vs_old = evaluate(params, hof, args.eval_maps)
                ev_bot, ev_old = vs_bot, vs_old
                line = f" | EVAL vs_bot {vs_bot:.2f} vs_old {vs_old:.2f}"
                write_export(os.path.join(args.save_dir, "export_latest"), params)
                if vs_bot + vs_old > best_score:
                    best_score = vs_bot + vs_old
                    write_export(os.path.join(args.save_dir, "export_best"), params)
                    line += " *best"
            if update % args.save_every == 0 or update == total_updates:
                torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), update=update, steps=steps_done,
                                best=best_score, hof=hof), last_path)

            def rate(kind, res):
                ep = [e for e in recent if kind in ("all", e["opp"])]
                return sum(e["result"] == res for e in ep) / len(ep) if ep else float("nan")

            dt = time.time() - t0
            print(f"upd {update:5d}/{total_updates} | {steps_done / 1e6:6.2f}M steps | {N * T / dt:7.0f} sps | "
                  f"win {rate('all', 'win'):.2f} loss {rate('all', 'loss'):.2f} draw {rate('all', 'draw'):.2f} "
                  f"(bot win {rate('bot', 'win'):.2f}) | ep_len {np.mean([e['steps'] for e in recent]) if recent else 0:5.0f} "
                  f"| pg {np.mean(stats['pg']):+.3f} v {np.mean(stats['v']):.3f} ent {np.mean(stats['ent']):.2f} "
                  f"kl {np.mean(stats['kl']):.4f} | roll {t_roll:.1f}s upd {dt - t_roll:.1f}s{line}", flush=True)
            append_metrics(args.save_dir, dict(
                update=update, steps=steps_done, sps=round(N * T / dt), win=rate('all', 'win'), loss=rate('all', 'loss'),
                draw=rate('all', 'draw'), bot_win=rate('bot', 'win'),
                ep_len=np.mean([e['steps'] for e in recent]) if recent else "",
                pg=np.mean(stats['pg']), v=np.mean(stats['v']), ent=np.mean(stats['ent']), kl=np.mean(stats['kl']),
                eval_vs_bot=ev_bot, eval_vs_old=ev_old, time=round(time.time())))
            if ev_bot != "":
                make_bundle(args.save_dir)  # always-fresh hand-off folder on Drive
    finally:
        env.close()

    write_export(os.path.join(args.save_dir, "export_latest"), export_params(net))
    make_bundle(args.save_dir)
    print(f"bundle for review: {os.path.join(args.save_dir, 'bundle')}")
    print(f"done. duel it: PYTHONPATH=../playground/python python3 ../playground/python/referee.py {args.save_dir}/export_best/agent.py rule_based")


if __name__ == "__main__":
    main()
