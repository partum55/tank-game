"""
Training-only map variety for the GPU arena.

The official maps (Rust mapgen: 2-5 mirrored wall pairs, spawns on the two sides) are only ONE kind of arena. Mixing in other
kinds teaches movement that does not depend on that particular layout:
  open     no obstacles at all            - pure aiming / chasing / dodging (an easy curriculum start)
  sparse   1-2 wall pairs                 - mostly open
  dense    5-7 wall pairs + centre block  - corridors, tight turns, hiding
  close    spawns near the centre         - immediate fights
  free     spawns anywhere, any heading   - no habit of "enemy is to my right"
All maps keep the official guarantees: point-symmetric (fair), every gap >= 3 tank radii (always passable), spawns clear of walls.
Evaluation and duels keep using the official Rust maps, so a harmful mix would show up as a gap in `EVAL`.
"""
import math
import random

W, H, RADIUS = 1200.0, 800.0, 18.0
CLEARANCE = 3.0 * RADIUS
KINDS = ("open", "sparse", "dense", "close", "free")


def _overlaps(a, b, m):
    return a[0] - m < b[0] + b[2] and a[0] + a[2] + m > b[0] and a[1] - m < b[1] + b[3] and a[1] + a[3] + m > b[1]


def _near(o, p, m):
    return o[0] - m < p[0] < o[0] + o[2] + m and o[1] - m < p[1] < o[1] + o[3] + m


def _mirror(o):
    return (W - o[0] - o[2], H - o[1] - o[3], o[2], o[3])


def _fits(o, twin, existing, spawns):
    if o[0] - CLEARANCE < 0 or o[1] - CLEARANCE < 0 or o[0] + o[2] + CLEARANCE > W or o[1] + o[3] + CLEARANCE > H:
        return False
    if any(_overlaps(o, e, CLEARANCE) for e in existing):
        return False
    if twin and (_overlaps(o, twin, CLEARANCE) or any(_overlaps(twin, e, CLEARANCE) for e in existing)):
        return False
    return not any(_near(o, s, CLEARANCE) for s in spawns)


def synth_map(rng: random.Random, kind: str):
    """-> (rects [(x, y, w, h)], pos [[x1, y1], [x2, y2]], angles [a1, a2])"""
    if kind == "close":
        x1, y1 = rng.uniform(0.30 * W, 0.44 * W), rng.uniform(0.15 * H, 0.85 * H)
    elif kind == "free":
        x1, y1 = rng.uniform(0.06 * W, 0.46 * W), rng.uniform(0.08 * H, 0.92 * H)
    else:
        x1, y1 = rng.uniform(0.08 * W, 0.22 * W), rng.uniform(0.15 * H, 0.85 * H)
    spawns = [(x1, y1), (W - x1, H - y1)]
    if kind == "free" or kind == "close":
        a1 = rng.uniform(-math.pi, math.pi)
    else:
        a1 = rng.uniform(-math.pi / 2, math.pi / 2)
    a2 = a1 + math.pi
    a2 = a2 - 2 * math.pi if a2 > math.pi else a2

    pairs = {"open": 0, "sparse": rng.randint(1, 2), "dense": rng.randint(5, 7),
             "close": rng.randint(1, 4), "free": rng.randint(2, 5)}[kind]
    rects = []
    if kind in ("dense", "free", "close") and rng.random() < 0.5:
        bw, bh = rng.uniform(50, 120), rng.uniform(50, 120)
        block = (0.5 * W - 0.5 * bw, 0.5 * H - 0.5 * bh, bw, bh)
        if _fits(block, None, rects, spawns):
            rects.append(block)
    lo = 0.12 * W if kind in ("close", "free") else 0.25 * W
    for _ in range(pairs):
        for _try in range(100):
            length, thick = rng.uniform(70 if kind == "dense" else 90, 220 if kind == "dense" else 280), rng.uniform(20, 40)
            ow, oh = (length, thick) if rng.random() < 0.5 else (thick, length)
            x = rng.uniform(lo, max(lo, (1 - lo / W) * W - ow))
            y = rng.uniform(CLEARANCE, max(CLEARANCE, H - CLEARANCE - oh))
            wall = (x, y, ow, oh)
            twin = _mirror(wall)
            if _fits(wall, twin, rects, spawns):
                rects += [wall, twin]
                break
    return rects, [list(spawns[0]), list(spawns[1])], [a1, a2]


def build_mix(n: int, seed: int = 0):
    """n synthetic maps cycling through all kinds."""
    rng = random.Random(seed)
    return [synth_map(rng, KINDS[i % len(KINDS)]) for i in range(n)]
