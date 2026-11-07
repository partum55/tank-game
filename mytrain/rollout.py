"""
One PPO rollout (policy -> arena, T steps) over static tensors, eager or as a CUDA graph.

Why: the GPU arena step is ~1.7k tiny kernels, so on an A100 it is bound by the CPU launching them (~37 ms per step whether
there are 1k or 65k arenas). A CUDA graph records the whole step once and replays it with a single launch.

The rules for capture: no host syncs (checked by hand in gpu_arena/gpu_selfplay), every piece of state lives in a tensor that keeps
its address (`_static_env_step` copies the freshly computed state back into the original tensors), randomness comes from the CUDA
generator (graph-safe), and the step index is a device tensor (`t`), not a Python int.
Anything going wrong while capturing falls back to the eager loop, so a training run never fails because of the graph.
"""
import time

import torch

ARENA_STATE = ("rects", "pos", "vel", "body", "turret", "speed", "hp", "alive", "cooldown", "steps", "bpos", "bvel", "bowner",
               "bactive", "mem_pos", "mem_turret", "mem_hp", "mem_unseen", "blocked", "prev_en", "prev_vis", "blocked_move")
ENV_STATE = ("side", "opp", "opp_h", "phi", "stats")


def static_env_step(env, act):
    """env.step(act), but every state tensor of the env/arena keeps its identity (needed to replay a captured graph)."""
    arena = env.arena
    refs = [(arena, n, getattr(arena, n)) for n in ARENA_STATE] + [(env, n, getattr(env, n)) for n in ENV_STATE]
    old_ob = env.ob
    result = env.step(act)
    for owner, name, ref in refs:
        new = getattr(owner, name)
        if new is not ref:
            ref.copy_(new)
            setattr(owner, name, ref)
    for name, ref in old_ob.__dict__.items():          # the Obs object too: same tensors, new values
        ref.copy_(getattr(env.ob, name))
    env.ob = old_ob
    return result


class Rollout:
    def __init__(self, net, env, T, hidden, device, reward_scale, feature_dim=31, graph=False, static_env=False):
        self.net, self.env, self.T, self.device, self.scale = net, env, T, device, reward_scale
        self.static_env = graph or static_env
        N = env.E if hasattr(env, "E") else env.vec.num_envs
        self.N = N
        z = lambda *s: torch.zeros(*s, device=device)
        self.obs = env.reset().clone()                 # current learner observation (N, F)
        self.h = z(N, hidden)
        self.first = torch.ones(N, device=device)
        self.t = torch.zeros(1, dtype=torch.long, device=device)
        self.obs_buf, self.move_buf = z(T, N, feature_dim), z(T, N, 3)
        self.first_buf, self.shoot_buf, self.logp_buf, self.val_buf, self.rew_buf, self.done_buf = (z(T, N) for _ in range(6))
        self.graph = None
        self.graph_note = "eager"
        if graph:
            self._try_capture()

    # ------------------------------------------------------------------ the step
    def _one_step(self):
        net = self.net
        with torch.no_grad():
            out, val, h_new = net.step(self.obs, self.h, self.first)
            move = out[:, :3] + net.log_std.exp() * torch.randn_like(out[:, :3])
            shoot = torch.bernoulli(torch.sigmoid(out[:, 3]))
            logp, _ = net.logp_entropy(out, move, shoot)
            act = torch.cat([move.clamp(-1, 1), (shoot * 2 - 1).unsqueeze(1)], dim=1)
            step = static_env_step if self.static_env else (lambda e, a: e.step(a))
            nxt, rew, done = step(self.env, act)
            t = self.t
            for buf, x in ((self.obs_buf, self.obs), (self.move_buf, move), (self.first_buf, self.first),
                           (self.shoot_buf, shoot), (self.logp_buf, logp), (self.val_buf, val),
                           (self.rew_buf, rew * self.scale), (self.done_buf, done)):
                buf.index_copy_(0, t, x.unsqueeze(0))
            self.h.copy_(h_new)
            self.obs.copy_(nxt)
            self.first.copy_(done)
            t.add_(1)

    # ------------------------------------------------------------------ graph
    def _try_capture(self):
        """Capture one step; benchmark it against the eager loop; keep it only if it works and is faster."""
        try:
            if self.device.type != "cuda":
                raise RuntimeError("CUDA graphs need a CUDA device")
            sync = torch.cuda.synchronize
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):                       # warm-up (cuBLAS/cuDNN handles, allocator) as PyTorch requires
                for _ in range(3):
                    self._one_step()
            torch.cuda.current_stream().wait_stream(side)
            sync()
            self.static_env = False                             # honest eager timing: the plain step, no state copy-backs
            t0 = time.time()
            for _ in range(10):
                self._one_step()
            sync()
            eager = (time.time() - t0) / 10
            self.static_env = True                              # capture with the identity-preserving step
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._one_step()
            sync()
            before = self.env.arena.steps.clone()
            t0 = time.time()
            for _ in range(10):
                graph.replay()
            sync()
            replay = (time.time() - t0) / 10
            progressed = bool((self.env.arena.steps != before).any())   # the replay really advanced the game
            finite = bool(torch.isfinite(self.obs).all() and torch.isfinite(self.rew_buf).all())
            if not (progressed and finite):
                raise RuntimeError(f"graph replay did not advance the game correctly (progressed={progressed}, finite={finite})")
            self.eager_ms, self.replay_ms = eager * 1e3, replay * 1e3
            if replay < eager:
                self.graph = graph
                self.graph_note = f"CUDA graph ON: {eager * 1e3:.1f} ms/step eager -> {replay * 1e3:.1f} ms/step replayed ({eager / replay:.1f}x)"
            else:
                self.graph_note = f"CUDA graph captured but not faster ({eager * 1e3:.1f} -> {replay * 1e3:.1f} ms/step): using the eager loop"
        except Exception as err:  # fall back, never fail the run because of the graph
            self.graph = None
            self.static_env = False
            self.graph_note = f"CUDA graph OFF (capture failed: {type(err).__name__}: {str(err)[:200]}) - using the eager loop"

    # ------------------------------------------------------------------ one rollout
    def run(self):
        self.t.zero_()
        for _ in range(self.T):
            if self.graph is not None:
                self.graph.replay()
            else:
                self._one_step()
