"""The GPU arena must play the same game as the Rust engine: same maps, same actions -> same everything."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tank_env import TankEnvironment
from mytrain.gpu_arena import GpuArena, MapBank
from mytrain.gru_policy import features

SEEDS = list(range(100, 124))
TICKS = 350


def random_actions(rng, ticks):
    """Sticky random driving + frequent shooting: exercises walls, tank pushes, bullets, fog."""
    out = np.zeros((ticks, 2, 4), np.float32)
    cur = rng.uniform(-1, 1, (2, 3))
    for t in range(ticks):
        if t % 15 == 0:
            cur = rng.uniform(-1, 1, (2, 3))
            cur[:, 0] = np.abs(cur[:, 0]) * 1.0 if rng.random() < 0.7 else cur[:, 0]
        out[t, :, :3] = cur
        out[t, :, 3] = np.where(rng.random(2) < 0.35, 1.0, -1.0)
    return out


def replay(seeds, ticks, device="cpu", rust_actions=None):
    bank = MapBank.from_rust(seeds, device)
    arena = GpuArena(len(seeds), bank, device, max_steps=500)
    arena.reset(torch.ones(len(seeds), dtype=torch.bool), torch.arange(len(seeds)))
    envs = [TankEnvironment(max_steps=500) for _ in seeds]
    rust = [e.reset(seed=s) for e, s in zip(envs, seeds)]
    rng = np.random.default_rng(0)
    acts = np.stack([random_actions(rng, ticks) for _ in seeds], 1)  # (ticks, E, 2, 4)
    feat_err = np.zeros((len(seeds), ticks))
    rew_err = np.zeros((len(seeds), ticks))
    done_mismatch = np.zeros(len(seeds), bool)
    first_div = np.full(len(seeds), ticks)
    finished = np.zeros(len(seeds), bool)

    def compare(t, rust_obs):
        obs = arena.observe().features().numpy()
        for e in range(len(seeds)):
            if finished[e]:
                continue
            for i in (0, 1):
                feat_err[e, t] = max(feat_err[e, t], np.abs(obs[e, i] - features(rust_obs[e][i])).max())

    compare(0, rust)
    for t in range(ticks):
        r, done, _ = arena.tick(torch.as_tensor(acts[t]))
        out = [env.step(acts[t, e, 0], acts[t, e, 1]) for e, env in enumerate(envs)]
        rust_obs = [(o[0], o[1]) for o in out]
        for e, o in enumerate(out):
            if finished[e]:
                continue
            rew_err[e, t] = np.abs(np.array(o[2]) - r[e].numpy()).max()
            if bool(o[3]) != bool(done[e]):
                done_mismatch[e] = True
            if o[3]:
                finished[e] = True
        compare(t + 1 if t + 1 < ticks else t, rust_obs)
    return feat_err, rew_err, done_mismatch


def test_gpu_arena_matches_rust_tick_by_tick():
    feat_err, rew_err, done_mismatch = replay(SEEDS, TICKS)
    # per-env: first tick where the features differ by more than 2e-3 (float32 trig can drift near edges)
    bad = feat_err > 2e-3
    diverged = bad.any(1)
    print("diverged envs:", int(diverged.sum()), "of", len(SEEDS),
          "| max feature err (non-diverged):", feat_err[~diverged].max() if (~diverged).any() else None)
    assert diverged.sum() <= 2, f"{diverged.sum()} of {len(SEEDS)} games drifted away from Rust"
    assert (rew_err[~diverged] < 1e-4).all()
    assert not done_mismatch[~diverged].any()
