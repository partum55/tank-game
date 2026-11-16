"""
GPU re-implementation of the tank arena for TRAINING only (thousands of arenas per tensor op).

The official game stays the Rust engine (referee, evaluation, duels). This module copies its tick
order and formulas 1:1 - tests/test_gpu_parity.py replays random games on both and compares every
tick - so what is learned here transfers. Maps and spawns are NOT regenerated here: they come from a
bank built by the Rust map generator, so the map distribution is exactly the official one.

Tick order (arena.rs): reload timers -> movement -> shooting -> tank/tank push -> obstacle push ->
border clamp -> bullets (obstacles, tank hits, border) -> fog memory -> win/draw -> engine reward.

Known, deliberate difference: when two bullets hit the same tank in the same tick the Rust engine
counts them one by one (a bullet after the killing one passes through); here all of them count.
Needs a tank dead at 4 hits to matter, so it only touches the very last tick of an episode.
"""
import math

import numpy as np
import torch

from .gru_policy import NO_SHOT, RELOAD_TICKS, SHOT_THETA, THREAT_SCALE, THREAT_TICKS

PI = math.pi
TWO_PI = 2.0 * math.pi
INF = float("inf")

# ArenaConfig::default() of the Rust engine (checked by the parity test through behaviour)
W, H = 1200.0, 800.0
RADIUS, MAX_SPEED, MAX_REVERSE, ACCEL = 18.0, 4.5, 2.5, 0.35
TURN_SPEED, TURRET_SPEED = 0.05, 0.07
BULLET_SPEED, DAMAGE, RELOAD = 9.0, 25.0, 20
MAX_AMMO = 20                       # rules v1.2: shots per tank per match
HULL_L, HULL_W = 40.0, 26.0         # rules v1.2: the tank is a HULL_L x HULL_W rectangle (RADIUS stays the map generator's unit)
HL, HW = HULL_L / 2, HULL_W / 2
MUZZLE = HL + 5.0
DIAG = math.sqrt(W * W + H * H)
RAY_ANGLES = [-0.75 * PI, -0.5 * PI, -0.25 * PI, 0.0, 0.25 * PI, 0.5 * PI, 0.75 * PI, PI]
PAD_RECT = (-1.0e6, -1.0e6, 0.0, 0.0)  # never touches anything

IN_PROGRESS, T1_WIN, T2_WIN, DRAW_TIE, DRAW_TIMEOUT = 0, 1, 2, 3, 4


def wrap(a):
    return torch.where(a > PI, a - TWO_PI, torch.where(a < -PI, a + TWO_PI, a))


def ray_rect(ox, oy, dx, dy, rx, ry, rw, rh):
    """Distance along the ray to the rectangle (inf = no hit). Mirrors math.rs::ray_rect_intersect."""
    t_min = torch.zeros_like(ox + dx + rx)
    t_max = torch.full_like(t_min, INF)
    fail = torch.zeros_like(t_min, dtype=torch.bool)
    for o, d, lo, size in ((ox, dx, rx, rw), (oy, dy, ry, rh)):
        hi = lo + size
        moving = d.abs() > 1e-6
        sd = torch.where(moving, d, torch.ones_like(d))
        t1, t2 = (lo - o) / sd, (hi - o) / sd
        t_min = torch.where(moving, torch.maximum(t_min, torch.minimum(t1, t2)), t_min)
        t_max = torch.where(moving, torch.minimum(t_max, torch.maximum(t1, t2)), t_max)
        fail = fail | (~moving & ((o < lo) | (o > hi)))
    fail = fail | (t_min > t_max) | (t_max < 0)
    return torch.where(fail, torch.full_like(t_min, INF), t_min)


def segment_hits_rect(p1, p2, rects):
    """p1,p2 (...,2); rects (...,K,4) broadcastable -> bool (...,) any rect crossed by the segment."""
    d = p2 - p1
    ln = d.norm(dim=-1)
    dirn = d / ln.clamp_min(1e-12).unsqueeze(-1)
    t = ray_rect(p1[..., 0:1], p1[..., 1:2], dirn[..., 0:1], dirn[..., 1:2],
                 rects[..., 0], rects[..., 1], rects[..., 2], rects[..., 3])
    return ((t <= ln.unsqueeze(-1)) & (ln.unsqueeze(-1) >= 1e-6)).any(-1)


def ray_circle(origin, d, center, r):
    """Mirrors math.rs::ray_circle_intersect. Returns (t, hit)."""
    m = origin - center
    b = (m * d).sum(-1)
    c = (m * m).sum(-1) - r * r
    discr = b * b - c
    s = discr.clamp_min(0).sqrt()
    t = -b - s
    t = torch.where(t < 0, -b + s, t)
    hit = ~((c > 0) & (b > 0)) & (discr >= 0) & (t >= 0)
    return t, hit


def ray_obb(ox, oy, dx, dy, cx, cy, ang):
    """Distance along the ray to the tank hull centred at (cx, cy) heading `ang` (inf = no hit). Mirrors math.rs::ray_obb_intersect."""
    c, s = torch.cos(ang), torch.sin(ang)
    lx, ly = ox - cx, oy - cy
    return ray_rect(lx * c + ly * s, -lx * s + ly * c, dx * c + dy * s, -dx * s + dy * c, -HL, -HW, 2 * HL, 2 * HW)


def obb_mtv(c1, a1, c2, a2, hl2, hw2):
    """SAT between the tank hull (centre c1 (...,2), heading a1 (...)) and a second box (centre c2, heading a2, half extents hl2, hw2).
    Returns (push (...,2), hit (...)): the smallest translation that moves box 1 out of box 2. Mirrors math.rs::obb_obb_mtv
    (same axis order u1, v1, u2, v2 and strict '<', so ties pick the same axis)."""
    s1, k1, s2, k2 = torch.sin(a1), torch.cos(a1), torch.sin(a2), torch.cos(a2)
    u1, v1 = torch.stack([k1, s1], -1), torch.stack([-s1, k1], -1)
    u2, v2 = torch.stack([k2, s2], -1), torch.stack([-s2, k2], -1)
    delta = c1 - c2
    best = torch.full_like(a1, INF)
    push = torch.zeros_like(delta)
    separated = torch.zeros_like(a1, dtype=torch.bool)
    for axis in (u1, v1, u2, v2):
        r1 = (HL * (u1 * axis).sum(-1)).abs() + (HW * (v1 * axis).sum(-1)).abs()
        r2 = (hl2 * (u2 * axis).sum(-1)).abs() + (hw2 * (v2 * axis).sum(-1)).abs()
        dist = (delta * axis).sum(-1)
        overlap = r1 + r2 - dist.abs()
        separated = separated | (overlap <= 0)
        better = overlap < best
        sgn = torch.where(dist >= 0, torch.ones_like(dist), -torch.ones_like(dist))
        push = torch.where(better.unsqueeze(-1), axis * (sgn * overlap).unsqueeze(-1), push)
        best = torch.where(better, overlap, best)
    hit = ~separated
    return push * hit.unsqueeze(-1), hit


class MapBank:
    """Maps + spawns generated by the Rust engine (exact official distribution)."""

    def __init__(self, rects, pos, angle, device):
        self.rects = rects.to(device)      # (M,K,4)
        self.pos = pos.to(device)          # (M,2,2)  tank1 / tank2 spawn
        self.angle = angle.to(device)      # (M,2)
        self.size = rects.shape[0]
        self.k = rects.shape[1]

    @classmethod
    def from_rust(cls, seeds, device, extra_maps=()):
        """Official maps from the Rust generator (+ optional training-only `extra_maps`, see mapmix.py)."""
        from tank_env import TankEnvironment

        env = TankEnvironment(max_steps=10)
        maps = []
        for s in seeds:
            o1, o2 = env.reset(seed=int(s))
            rects = [[x * W, y * H, w * W, h * H] for x, y, w, h in o1["obstacles"]]
            pos = [[o1["x"] * W, o1["y"] * H], [o2["x"] * W, o2["y"] * H]]
            maps.append((rects, pos, [o1["body_angle"], o2["body_angle"]]))
        return cls.from_maps(maps + list(extra_maps), device)

    @classmethod
    def from_maps(cls, maps, device):
        """maps: [(rects [(x,y,w,h)], pos [[x1,y1],[x2,y2]], angles [a1,a2])]"""
        k = max(max(len(m[0]) for m in maps), 1)
        rects = np.tile(np.array(PAD_RECT, np.float32), (len(maps), k, 1))
        for i, m in enumerate(maps):
            if m[0]:
                rects[i, :len(m[0])] = np.array(m[0], np.float32)
        return cls(torch.as_tensor(rects), torch.as_tensor(np.array([m[1] for m in maps], np.float32)),
                   torch.as_tensor(np.array([m[2] for m in maps], np.float32)), device)


class Obs:
    """Everything a tank can see, per observer (E,2,...). Raw fields mirror the Rust telemetry."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def features(self):
        """The 36 inputs of the model (identical to mytrain.gru_policy.FeatureExtractor on the Rust telemetry)."""
        sc = lambda a: (torch.sin(a), torch.cos(a))
        s1, c1 = sc(self.body)
        s2, c2 = sc(self.turret - self.body)
        s3, c3 = sc(self.angle_to)
        s4, c4 = sc(self.enemy_turret - self.body)
        cols = [self.speed, self.vx, self.vy, s1, c1, s2, c2, self.hp / 100.0, self.reload,
                self.rel_x, self.rel_y, self.dist, s3, c3, s4, c4, self.enemy_hp / 100.0,
                self.visible.float(), self.last_seen]
        extra = torch.stack([self.ev_fwd, self.ev_side, torch.sin(self.lead_err), torch.cos(self.lead_err)], -1)
        v2 = torch.stack([self.ammo, self.threat[..., 0], self.threat[..., 1], self.threat[..., 2], self.shot_p], -1)
        return torch.cat([torch.stack(cols, -1), self.lidar, extra, v2], -1)


class GpuArena:
    def __init__(self, num_envs, bank: MapBank, device, max_steps=1500, bullets=24):
        E, B, K = num_envs, bullets, bank.k
        self.E, self.B, self.K, self.device, self.max_steps, self.bank = E, B, K, device, max_steps, bank
        f = lambda *s: torch.zeros(*s, device=device)
        self.rects = f(E, K, 4)
        self.pos, self.vel = f(E, 2, 2), f(E, 2, 2)
        self.body, self.turret, self.speed, self.hp = f(E, 2), f(E, 2), f(E, 2), f(E, 2)
        self.alive = torch.ones(E, 2, dtype=torch.bool, device=device)
        self.cooldown = torch.zeros(E, 2, dtype=torch.long, device=device)
        self.steps = torch.zeros(E, dtype=torch.long, device=device)
        self.bpos, self.bvel = f(E, B, 2), f(E, B, 2)
        self.bowner = torch.zeros(E, B, dtype=torch.long, device=device)
        self.bactive = torch.zeros(E, B, dtype=torch.bool, device=device)
        self.mem_pos, self.mem_turret, self.mem_hp = f(E, 2, 2), f(E, 2), f(E, 2)
        self.mem_unseen = torch.zeros(E, 2, dtype=torch.long, device=device)
        self.ray_angles = torch.tensor(RAY_ANGLES, device=device)
        self.slot_ids = torch.arange(B, device=device)
        self.blocked = torch.zeros(E, 2, dtype=torch.bool, device=device)
        self.prev_en = f(E, 2, 2)                    # enemy position at the previous tick (for its velocity)
        self.prev_vis = torch.zeros(E, 2, dtype=torch.bool, device=device)
        self.blocked_move = torch.zeros(E, 2, dtype=torch.bool, device=device)   # tank commanded to move but stuck (wall / tank)
        self.ammo = torch.full((E, 2), MAX_AMMO, dtype=torch.long, device=device)
        self.shot_age = torch.full((E, 2), NO_SHOT, dtype=torch.long, device=device)   # ticks since tank i last spawned a bullet
        # events of the LAST tick, per tank (E,2): shots fired, shots that ended on a wall / the border / a blocked muzzle, hits TAKEN
        self.last_fired, self.last_missed, self.last_hits = f(E, 2), f(E, 2), f(E, 2)
        self.reset(torch.ones(E, dtype=torch.bool, device=device), self.random_maps())

    def random_maps(self, n=None):
        return torch.randint(0, self.bank.size, (n or self.E,), device=self.device)

    # ------------------------------------------------------------------ reset
    def reset(self, mask, map_idx):
        """Start a new episode in the envs where `mask` is true, on bank maps `map_idx` (E,)."""
        m = mask
        m1, m2 = m[:, None], m[:, None, None]
        b = self.bank
        self.rects = torch.where(m2, b.rects[map_idx], self.rects)
        self.pos = torch.where(m2, b.pos[map_idx], self.pos)
        self.body = torch.where(m1, b.angle[map_idx], self.body)
        self.turret = torch.where(m1, b.angle[map_idx], self.turret)
        self.vel = torch.where(m2, torch.zeros_like(self.vel), self.vel)
        self.speed = torch.where(m1, torch.zeros_like(self.speed), self.speed)
        self.hp = torch.where(m1, torch.full_like(self.hp, 100.0), self.hp)
        self.alive = torch.where(m1, torch.ones_like(self.alive), self.alive)
        self.cooldown = torch.where(m1, torch.zeros_like(self.cooldown), self.cooldown)
        self.ammo = torch.where(m1, torch.full_like(self.ammo, MAX_AMMO), self.ammo)
        self.shot_age = torch.where(m1, torch.full_like(self.shot_age, NO_SHOT), self.shot_age)
        self.steps = torch.where(m, torch.zeros_like(self.steps), self.steps)
        self.bactive = torch.where(m1, torch.zeros_like(self.bactive), self.bactive)
        # at the start both tanks know where the enemy is
        self.mem_pos = torch.where(m2, self.pos.flip(1), self.mem_pos)
        self.mem_turret = torch.where(m1, self.turret.flip(1), self.mem_turret)
        self.mem_hp = torch.where(m1, self.hp.flip(1), self.mem_hp)
        self.mem_unseen = torch.where(m1, torch.zeros_like(self.mem_unseen), self.mem_unseen)
        self.prev_vis = torch.where(m1, torch.zeros_like(self.prev_vis), self.prev_vis)
        self.prev_en = torch.where(m2, self.pos.flip(1), self.prev_en)
        self.blocked_move = torch.where(m1, torch.zeros_like(self.blocked_move), self.blocked_move)
        self._update_los()

    def _update_los(self):
        """blocked[:, i]: a wall is between observer i and its enemy (segment between tank centres)."""
        p = self.pos
        b0 = segment_hits_rect(p[:, 0], p[:, 1], self.rects)
        b1 = segment_hits_rect(p[:, 1], p[:, 0], self.rects)
        self.blocked = torch.stack([b0, b1], 1)

    def visible(self):
        return self.alive.flip(1) & ~self.blocked

    # ------------------------------------------------------------------ one physics tick
    def tick(self, act):
        """act (E,2,4): thrust, turn, turret_turn, shoot(+-1). Returns engine rewards (E,2), done (E,), outcome (E,)."""
        E, dev = self.E, self.device
        self.steps = self.steps + 1
        self.cooldown = (self.cooldown - 1).clamp_min(0)
        self.prev_en = self.pos.flip(1)          # enemy position / visibility as last observed
        self.prev_vis = self.visible()
        pos_before = self.pos
        alive = self.alive
        thrust, turn, tt = (act[..., i].clamp(-1, 1) for i in range(3))
        shoot = act[..., 3] > 0.5

        # movement (dead tanks stand still)
        body = wrap(self.body + turn * TURN_SPEED)
        turret = wrap(self.turret + tt * TURRET_SPEED)
        target = torch.where(thrust >= 0, thrust * MAX_SPEED, thrust * MAX_REVERSE)
        speed = self.speed + (target - self.speed) * ACCEL
        vel = torch.stack([torch.cos(body), torch.sin(body)], -1) * speed.unsqueeze(-1)
        a2 = alive.unsqueeze(-1)
        self.body = torch.where(alive, body, self.body)
        self.turret = torch.where(alive, turret, self.turret)
        self.speed = torch.where(alive, speed, torch.zeros_like(speed))
        self.vel = torch.where(a2, vel, torch.zeros_like(vel))
        self.pos = torch.where(a2, self.pos + vel, self.pos)

        # shooting (tank 1 then tank 2, both from post-movement positions)
        can = shoot & (self.cooldown == 0) & alive & (self.ammo > 0)
        wasted, spawned = [], []   # wasted: shots that burn ammo against a wall at the muzzle (counted as misses, like the Rust engine)
        for i in (0, 1):
            d = torch.stack([torch.cos(self.turret[:, i]), torch.sin(self.turret[:, i])], -1)
            spawn = self.pos[:, i] + d * MUZZLE
            blocked = segment_hits_rect(self.pos[:, i], spawn, self.rects)
            wasted.append(can[:, i] & blocked)
            ok = can[:, i] & ~blocked
            slot = (~self.bactive).float().argmax(1)
            one = (self.slot_ids == slot[:, None]) & ok[:, None]   # no one_hot: nothing may sync the host (CUDA graphs)
            self.bpos = torch.where(one.unsqueeze(-1), spawn[:, None], self.bpos)
            self.bvel = torch.where(one.unsqueeze(-1), (d * BULLET_SPEED)[:, None], self.bvel)
            self.bowner = torch.where(one, torch.full_like(self.bowner, i), self.bowner)
            self.bactive = self.bactive | one
            spawned.append(one)
        self.cooldown = torch.where(can, torch.full_like(self.cooldown, RELOAD), self.cooldown)
        self.ammo = self.ammo - can.long()
        wasted = torch.stack(wasted, 1)

        # tank / tank push (oriented hulls, SAT; split symmetrically)
        push, _ = obb_mtv(self.pos[:, 0], self.body[:, 0], self.pos[:, 1], self.body[:, 1], HL, HW)
        self.pos = torch.stack([self.pos[:, 0] + 0.5 * push, self.pos[:, 1] - 0.5 * push], 1)

        # obstacle push (obstacle by obstacle, as in Rust; both tanks at once - they don't interact here)
        for k in range(self.K):
            rx, ry, rw, rh = (self.rects[:, k, j].unsqueeze(1) for j in range(4))
            centre = torch.stack([rx + 0.5 * rw, ry + 0.5 * rh], -1).expand(-1, 2, -1)
            push, _ = obb_mtv(self.pos, self.body, centre, torch.zeros_like(self.body), 0.5 * rw, 0.5 * rh)
            self.pos = self.pos + push

        # border clamp: the rotated hull's axis-aligned extents (also removes the velocity component pointing into the wall)
        ca, sa = self.body.cos().abs(), self.body.sin().abs()
        ex, ey = ca * HL + sa * HW, sa * HL + ca * HW
        x, y, vx, vy = self.pos[..., 0], self.pos[..., 1], self.vel[..., 0], self.vel[..., 1]
        vx = torch.where(((x < ex) & (vx < 0)) | ((x > W - ex) & (vx > 0)), torch.zeros_like(vx), vx)
        vy = torch.where(((y < ey) & (vy < 0)) | ((y > H - ey) & (vy > 0)), torch.zeros_like(vy), vy)
        self.pos = torch.stack([torch.maximum(torch.minimum(x, W - ex), ex), torch.maximum(torch.minimum(y, H - ey), ey)], -1)
        self.vel = torch.stack([vx, vy], -1)
        moved = (self.pos - pos_before).norm(dim=-1)
        self.blocked_move = alive & (self.speed.abs() > 1.0) & (moved < 0.3 * self.speed.abs())   # driving into something

        # bullets
        nxt = self.bpos + self.bvel
        travel = self.bvel.norm(dim=-1)
        dirn = self.bvel / travel.clamp_min(1e-12).unsqueeze(-1)
        on_wall = self.bactive & segment_hits_rect(self.bpos, nxt, self.rects[:, None])
        act_b = self.bactive & ~on_wall
        hits = []
        for i in (0, 1):
            t = ray_obb(self.bpos[..., 0], self.bpos[..., 1], dirn[..., 0], dirn[..., 1],
                        self.pos[:, i, 0:1], self.pos[:, i, 1:2], self.body[:, i, None])
            h = act_b & (self.bowner != i) & alive[:, i, None] & (t <= travel)
            act_b = act_b & ~h
            hits.append(h.sum(1))
        hits = torch.stack(hits, 1)  # hits taken by tank i
        out = (nxt[..., 0] <= 0) | (nxt[..., 0] >= W) | (nxt[..., 1] <= 0) | (nxt[..., 1] >= H)
        gone = on_wall | (act_b & out)       # bullets that ended on a wall / the border: a miss for their owner
        act_b = act_b & ~out
        missed = torch.stack([(gone & (self.bowner == i)).sum(1) for i in (0, 1)], 1) + wasted.long()
        self.last_fired, self.last_missed, self.last_hits = can.float(), missed.float(), hits.float()
        self.bpos = torch.where(act_b.unsqueeze(-1), nxt, self.bpos)
        self.bactive = act_b
        # a shot is only SEEN if its bullet survives the tick it was fired in (not a wall / point-blank hit): that is what the agent can observe
        seen_shot = torch.stack([(m & act_b).any(1) for m in spawned], 1)
        self.shot_age = torch.where(seen_shot, torch.zeros_like(self.shot_age), (self.shot_age + 1).clamp(max=NO_SHOT))
        self.hp = (self.hp - DAMAGE * hits).clamp_min(0)
        self.alive = self.alive & (self.hp > 0)

        # fog memory: what each tank last saw of its enemy
        self._update_los()
        sees = self.visible()
        s2 = sees.unsqueeze(-1)
        self.mem_pos = torch.where(s2, self.pos.flip(1), self.mem_pos)
        self.mem_turret = torch.where(sees, self.turret.flip(1), self.mem_turret)
        self.mem_hp = torch.where(sees, self.hp.flip(1), self.mem_hp)
        self.mem_unseen = torch.where(sees, torch.zeros_like(self.mem_unseen), self.mem_unseen + 1)

        # outcome
        a0, a1 = self.alive[:, 0], self.alive[:, 1]
        timeout = (self.steps >= self.max_steps) & a0 & a1
        outcome = torch.zeros(E, dtype=torch.long, device=dev)
        outcome = torch.where(~a0 & ~a1, torch.full_like(outcome, DRAW_TIE), outcome)
        outcome = torch.where(~a0 & a1, torch.full_like(outcome, T2_WIN), outcome)
        outcome = torch.where(a0 & ~a1, torch.full_like(outcome, T1_WIN), outcome)
        hp0, hp1 = self.hp[:, 0], self.hp[:, 1]
        by_hp = torch.where(hp0 > hp1, torch.full_like(outcome, T1_WIN),
                            torch.where(hp1 > hp0, torch.full_like(outcome, T2_WIN),
                                        torch.full_like(outcome, DRAW_TIMEOUT)))
        outcome = torch.where(timeout, by_hp, outcome)
        done = outcome > 0

        # engine reward (before the trainer's shaping)
        r = 0.01 + torch.stack([hits[:, 1] * 15.0 - hits[:, 0] * 10.0, hits[:, 0] * 15.0 - hits[:, 1] * 10.0], 1)
        win1, win2 = outcome == T1_WIN, outcome == T2_WIN
        drw = (outcome == DRAW_TIE) | (outcome == DRAW_TIMEOUT)
        r = r + torch.stack([100.0 * win1 - 100.0 * win2 - 10.0 * drw, 100.0 * win2 - 100.0 * win1 - 10.0 * drw], 1)
        return r, done, outcome

    # ------------------------------------------------------------------ what each tank sees
    def observe(self, lidar=True) -> Obs:
        E, dev = self.E, self.device
        me_pos, en_pos = self.pos, self.pos.flip(1)
        vis = self.visible()
        en_alive = self.alive.flip(1)
        reveal = vis | ~en_alive
        seen_pos = torch.where(reveal.unsqueeze(-1), en_pos, self.mem_pos)
        seen_turret = torch.where(reveal, self.turret.flip(1), self.mem_turret)
        seen_hp = torch.where(reveal, self.hp.flip(1), self.mem_hp)
        unseen = torch.where(reveal, torch.zeros_like(self.mem_unseen), self.mem_unseen)
        rel = seen_pos - me_pos
        angle_to = wrap(torch.atan2(rel[..., 1], rel[..., 0]) - self.body)

        # enemy velocity = its move between two consecutive ticks in which it was visible, in the own-heading frame
        v = torch.where((vis & self.prev_vis).unsqueeze(-1), en_pos - self.prev_en, torch.zeros_like(en_pos))
        c_b, s_b = torch.cos(self.body), torch.sin(self.body)
        ev_fwd = ((c_b * v[..., 0] + s_b * v[..., 1]) / MAX_SPEED).clamp(-2, 2)
        ev_side = ((-s_b * v[..., 0] + c_b * v[..., 1]) / MAX_SPEED).clamp(-2, 2)
        # where to aim: intercept of a bullet (constant speed) with the extrapolated enemy
        a = (v * v).sum(-1) - BULLET_SPEED * BULLET_SPEED
        a = a.clamp(max=-1e-6)
        p = seen_pos - me_pos
        b = 2.0 * (p * v).sum(-1)
        c0 = (p * p).sum(-1)
        t_hit = ((-b - (b * b - 4.0 * a * c0).clamp_min(0).sqrt()) / (2.0 * a)).clamp(0, 400)
        q = p + v * t_hit.unsqueeze(-1)
        lead_err = wrap(torch.atan2(q[..., 1], q[..., 0]) - self.turret)

        if lidar:
            # lidar: 8 rays, nearest of border / enemy tank / obstacle
            ang = wrap(self.body.unsqueeze(-1) + self.ray_angles)           # (E,2,8)
            dx, dy = torch.cos(ang), torch.sin(ang)
            ox, oy = me_pos[..., 0:1], me_pos[..., 1:2]
            t_wall = self._border_t(ox, oy, dx, dy)
            d = torch.stack([dx, dy], -1)
            t_en = ray_obb(ox, oy, dx, dy, en_pos[..., 0:1], en_pos[..., 1:2], self.body.flip(1).unsqueeze(-1))
            t_en = torch.where(en_alive.unsqueeze(-1), t_en, torch.full_like(t_en, INF))
            R = self.rects[:, None, None]                                    # (E,1,1,K,4)
            t_ob = ray_rect(ox.unsqueeze(-1), oy.unsqueeze(-1), dx.unsqueeze(-1), dy.unsqueeze(-1),
                            R[..., 0], R[..., 1], R[..., 2], R[..., 3]).min(-1).values
            lidar_t = (torch.minimum(torch.minimum(t_wall, t_en), t_ob).clamp_max(DIAG) / DIAG).clamp(0, 1)
        else:
            lidar_t = torch.zeros(E, 2, 8, device=dev)
        # the extra v1.2 inputs belong to a full observation only (the shaping potential does not need them)
        threat = self._threat() if lidar else torch.zeros(E, 2, 3, device=dev)
        shot_p = self._shot_prob(seen_pos, seen_turret, me_pos, rel.norm(dim=-1) / DIAG) if lidar else torch.zeros(E, 2, device=dev)

        return Obs(threat=threat, shot_p=shot_p,
            body=self.body, turret=self.turret, speed=self.speed / MAX_SPEED,
            vx=self.vel[..., 0] / MAX_SPEED, vy=self.vel[..., 1] / MAX_SPEED, hp=self.hp,
            reload=self.cooldown.float() / RELOAD, ammo=self.ammo.float() / MAX_AMMO, rel_x=rel[..., 0] / W, rel_y=rel[..., 1] / H,
            dist=rel.norm(dim=-1) / DIAG, angle_to=angle_to, enemy_turret=seen_turret, enemy_hp=seen_hp,
            visible=vis, last_seen=unseen.float() / self.max_steps, lidar=lidar_t,
            ev_fwd=ev_fwd, ev_side=ev_side, lead_err=lead_err)

    def _threat(self):
        """Per observer: the incoming enemy bullet whose closest approach is soonest -> (urgency, miss distance, side) (E,2,3).
        Mirrors gru_policy.bullet_threat."""
        p = self.bpos[:, None] - self.pos[:, :, None]                         # (E,2,B,2) bullet relative to the observer
        v = self.bvel[:, None].expand(-1, 2, -1, -1)
        theirs = self.bowner[:, None] != torch.arange(2, device=self.device)[None, :, None]
        pv = (p * v).sum(-1)
        tc = -pv / (v * v).sum(-1).clamp_min(1e-6)
        cand = self.bactive[:, None] & theirs & (pv < 0) & (tc <= THREAT_TICKS)
        idx = torch.where(cand, tc, torch.full_like(tc, INF)).argmin(-1, keepdim=True)    # (E,2,1)
        found = cand.any(-1)
        m = (p + v * tc.unsqueeze(-1)).gather(2, idx.unsqueeze(-1).expand(-1, -1, 1, 2)).squeeze(2)   # closest-approach offset (E,2,2)
        lateral = -torch.sin(self.body) * m[..., 0] + torch.cos(self.body) * m[..., 1]
        zero = torch.zeros_like(lateral)
        urgency = torch.where(found, 1.0 - tc.gather(-1, idx).squeeze(-1) / THREAT_TICKS, zero)
        miss = torch.where(found, (m.norm(dim=-1) / THREAT_SCALE).clamp(max=2.0), torch.full_like(zero, 2.0))
        side = torch.where(found, (lateral / THREAT_SCALE).clamp(-1.0, 1.0), zero)
        return torch.stack([urgency, miss, side], -1)

    def _shot_prob(self, seen_pos, seen_turret, me_pos, dist):
        """p(the enemy is shooting now) = sigmoid(theta . [1, |turret error towards us| / pi, reload readiness, distance]); see gru_policy."""
        err = wrap(seen_turret - torch.atan2(me_pos[..., 1] - seen_pos[..., 1], me_pos[..., 0] - seen_pos[..., 0])).abs() / PI
        ready = self.shot_age.flip(1).float().clamp(max=RELOAD_TICKS) / RELOAD_TICKS
        x = torch.stack([torch.ones_like(err), err, ready, dist], -1)
        return torch.sigmoid(x @ torch.tensor(SHOT_THETA, device=self.device))

    @staticmethod
    def _border_t(ox, oy, dx, dy):
        """math.rs::ray_bounds_intersect for an origin inside the arena: distance to the border."""
        t_min = torch.zeros_like(ox + dx)
        t_max = torch.full_like(t_min, INF)
        for o, d, size in ((ox, dx, W), (oy, dy, H)):
            moving = d.abs() > 1e-6
            sd = torch.where(moving, d, torch.ones_like(d))
            t1, t2 = (0.0 - o) / sd, (size - o) / sd
            t_min = torch.where(moving, torch.maximum(t_min, torch.minimum(t1, t2)), t_min)
            t_max = torch.where(moving, torch.minimum(t_max, torch.maximum(t1, t2)), t_max)
        res = torch.where(t_min > 0, t_min, torch.where(t_max > 0, t_max, torch.full_like(t_min, INF)))
        return torch.where((t_min > t_max) | (t_max < 0), torch.full_like(res, INF), res)
