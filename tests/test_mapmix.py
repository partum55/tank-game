import math

import numpy as np
import pytest

from mytrain.mapmix import CLEARANCE, H, KINDS, RADIUS, W, _near, _overlaps, build_mix, synth_map


def test_every_synthetic_map_is_fair_and_passable():
    maps = build_mix(600, seed=3)
    counts = {k: [] for k in KINDS}
    for i, (rects, pos, ang) in enumerate(maps):
        counts[KINDS[i % len(KINDS)]].append(len(rects))
        # point-symmetric spawns, opposite headings
        assert abs(pos[0][0] + pos[1][0] - W) < 1e-6 and abs(pos[0][1] + pos[1][1] - H) < 1e-6
        d = abs((ang[1] - ang[0] + math.pi) % (2 * math.pi) - math.pi)
        assert d < 1e-5 or abs(d - 2 * math.pi) < 1e-5 or abs((ang[1] - ang[0]) % (2 * math.pi) - math.pi) < 1e-5
        assert all(-math.pi - 1e-6 <= a <= math.pi + 1e-6 for a in ang)
        for r in rects:
            twin = (W - r[0] - r[2], H - r[1] - r[3], r[2], r[3])
            assert any(np.allclose(twin, q, atol=1e-6) for q in rects) or (np.allclose(twin, r, atol=1e-6))
            assert r[0] >= CLEARANCE and r[1] >= CLEARANCE and r[0] + r[2] <= W - CLEARANCE and r[1] + r[3] <= H - CLEARANCE
            assert not any(_near(r, p, CLEARANCE) for p in pos)
        for a in range(len(rects)):
            for b in range(a + 1, len(rects)):
                assert not _overlaps(rects[a], rects[b], CLEARANCE - 1e-3)
    assert max(counts["open"]) == 0                      # open arenas are really empty
    assert np.mean(counts["dense"]) > np.mean(counts["sparse"]) + 3
    assert all(RADIUS * 2 < W for _ in maps)


def test_gpu_arena_runs_on_mixed_maps_without_leaving_the_arena():
    torch = pytest.importorskip("torch")
    from mytrain.gpu_arena import GpuArena, MapBank

    bank = MapBank.from_maps(build_mix(50, 1), "cpu")
    E = 50
    arena = GpuArena(E, bank, "cpu", max_steps=200)
    arena.reset(torch.ones(E, dtype=torch.bool), torch.arange(E))
    g = torch.Generator().manual_seed(0)
    for _ in range(150):
        act = torch.rand(E, 2, 4, generator=g) * 2 - 1
        act[..., 3] = torch.where(act[..., 3] > 0, 1.0, -1.0)
        r, done, _ = arena.tick(act)
        assert torch.isfinite(r).all()
    p = arena.pos
    assert (p[..., 0] >= 17.9).all() and (p[..., 0] <= 1200 - 17.9).all() and (p[..., 1] >= 17.9).all() and (p[..., 1] <= 800 - 17.9).all()
    feats = arena.observe().features()
    assert feats.shape == (E, 2, 27) and torch.isfinite(feats).all()
