"""
Pre-match adaptation of an exported agent. NumPy only, deterministic: a fixed number of steps, no clock, no randomness.

The referee keeps an agent's process alive for the whole series and calls reset() before every match, so the agent can learn from the match
that just ended against THIS opponent. What it learns (every part can be switched off: adapt="move,shot,bc,rl"):

  move  the opponent's motion: ridge regression from its current velocity to its mean velocity over the next HORIZON ticks (prior: "keeps its
        velocity", which is what the network was trained with). The aim's lead point is computed with the predicted velocity.
  shot  the opponent's shooting: refit of the 4 numbers (theta) of the "enemy is shooting" input on the shots we saw it fire (prior: defaults).
  bc    copy what worked for the opponent: its actions are reconstructed from our telemetry (stretches where it was in sight), our network is run
        on the situation as the OPPONENT saw it, and the output layer is pulled towards those actions (more weight just before its hits on us).
  rl    our own experience: REINFORCE on the match we just played, with the training reward (hits, bad shots, outcome).

Only the output layer (w_out, b_out) changes, and never by more than HEAD_MAX_REL of its norm from the exported weights.
What it cannot know: how the match ended (the referee does not pass the last observation) - inferred from the tick count and the last HPs.
"""
import math

import numpy as np

try:
    from .gru_policy import (BULLET_SPEED, FEATURE_DIM, H, MAX_SPEED, RELOAD_TICKS, SHOT_THETA, W, FeatureExtractor, _sigmoid)
except ImportError:  # standalone export: tank_gru_policy.py and tank_adapt.py sit next to agent.py
    from tank_gru_policy import (BULLET_SPEED, FEATURE_DIM, H, MAX_SPEED, RELOAD_TICKS, SHOT_THETA, W, FeatureExtractor, _sigmoid)

PARTS = ("move", "shot", "bc", "rl")
MAX_STEPS = 1500                  # official match length
MIN_TICKS = 60                    # shorter records are not worth learning from
HULL_HALF = (20.0, 13.0)          # rules v1.2 hull: half length, half width
ACCEL, MAX_REVERSE, TURN_SPEED, TURRET_SPEED = 0.35, 2.5, 0.05, 0.07   # arena constants (ArenaConfig::default)
DIAG = math.hypot(W, H)
RAY_ANGLES = [-0.75 * math.pi, -0.5 * math.pi, -0.25 * math.pi, 0.0, 0.25 * math.pi, 0.5 * math.pi, 0.75 * math.pi, math.pi]
HORIZON = 25                      # ticks the motion model looks ahead (a bullet needs ~ dist / 9 ticks)
SAMPLE_CAP = 6000
RIDGE = 100.0                     # pseudo-samples of the "keeps its velocity" prior
SHOT_RIDGE = 5.0                  # pseudo-samples of the default shot model
HEAD_STEPS = 30
HEAD_MAX_REL = 0.05               # the output layer may move by at most 5% of its norm away from the exported weights
GAMMA = 0.997
HIT_WINDOW = 30                   # ticks before a hit on us that count as "what worked" for the opponent
SHOT_BLIND, SHOT_OFF, HULL_HALF_WIDTH = 1.0, 0.5, 13.0   # keep equal to selfplay_env.py


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


# ------------------------------------------------------------------ geometry for the lidar of the opponent's view
def _ray_rect(ox, oy, dx, dy, rx, ry, rw, rh):
    t_min, t_max = 0.0, math.inf
    for o, d, lo, size in ((ox, dx, rx, rw), (oy, dy, ry, rh)):
        if abs(d) > 1e-6:
            t1, t2 = (lo - o) / d, (lo + size - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_min, t_max = max(t_min, t1), min(t_max, t2)
            if t_min > t_max or t_max < 0.0:
                return math.inf
        elif o < lo or o > lo + size:
            return math.inf
    return t_min if t_min >= 0.0 else (t_max if t_max >= 0.0 else math.inf)


def _ray_border(ox, oy, dx, dy):
    t_min, t_max = 0.0, math.inf
    for o, d, size in ((ox, dx, W), (oy, dy, H)):
        if abs(d) > 1e-6:
            t1, t2 = (0.0 - o) / d, (size - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_min, t_max = max(t_min, t1), min(t_max, t2)
            if t_min > t_max or t_max < 0.0:
                return math.inf
    return t_min if t_min > 0.0 else (t_max if t_max > 0.0 else math.inf)


def lidar8(pos, body, rects, other_pos, other_body):
    """The 8 lidar rays of a tank at `pos` heading `body`: nearest of border / the other tank's hull / walls, / arena diagonal."""
    out = []
    oc, os_ = math.cos(other_body), math.sin(other_body)
    lx, ly = pos[0] - other_pos[0], pos[1] - other_pos[1]
    for rel in RAY_ANGLES:
        a = body + rel
        dx, dy = math.cos(a), math.sin(a)
        t = min(DIAG, _ray_border(pos[0], pos[1], dx, dy))
        t = min(t, _ray_rect(lx * oc + ly * os_, -lx * os_ + ly * oc, dx * oc + dy * os_, -dx * os_ + dy * oc,
                             -HULL_HALF[0], -HULL_HALF[1], 2 * HULL_HALF[0], 2 * HULL_HALF[1]))
        for rx, ry, rw, rh in rects:
            t = min(t, _ray_rect(pos[0], pos[1], dx, dy, rx, ry, rw, rh))
        out.append(min(max(t / DIAG, 0.0), 1.0))
    return out


def visible_runs(rows, min_len=3):
    """[(start, end)) of stretches of consecutive rows in which the enemy was in sight."""
    runs, start = [], None
    for i, r in enumerate(list(rows) + [None]):
        if r is not None and r["vis"]:
            start = i if start is None else start
        elif start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    return runs


def enemy_view(rows, k, live, shots_seen, rects):
    """What the OPPONENT's telemetry looked like at row k: the same dict the arena gives an agent. `live`: the enemy was in sight at rows k-1
    and k, so its pose and velocity are known; otherwise the (stale) last known pose is used and only the bullets are meaningful (warm-up)."""
    r = rows[k]
    ex, ey = r["epos"]
    ox, oy = r["pos"]
    eb = r["ebody"]
    v = (ex - rows[k - 1]["epos"][0], ey - rows[k - 1]["epos"][1]) if live else (0.0, 0.0)
    dx, dy = ox - ex, oy - ey
    return dict(
        x=ex / W, y=ey / H, vx=v[0] / MAX_SPEED, vy=v[1] / MAX_SPEED, speed=(v[0] * math.cos(eb) + v[1] * math.sin(eb)) / MAX_SPEED,
        body_angle=eb, turret_angle=r["eturret"], hp=r["ehp"], reload_cooldown=max(0, RELOAD_TICKS - r["age"]) / RELOAD_TICKS,
        ammo=max(0, 20 - shots_seen) / 20.0,
        enemy_rel_x=dx / W, enemy_rel_y=dy / H, enemy_dist=math.hypot(dx, dy) / DIAG, enemy_angle_to=_wrap(math.atan2(dy, dx) - eb),
        enemy_turret_angle=r["turret"], enemy_body_angle=r["body"], enemy_hp=r["hp"], enemy_visible=bool(r["vis"]), enemy_last_seen=0.0,
        lidar_rays=lidar8((ex, ey), eb, rects, (ox, oy), r["body"]),
        bullets=[((bx - ex) / W, (by - ey) / H, bvx / BULLET_SPEED, bvy / BULLET_SPEED, 1.0 - mine) for bx, by, bvx, bvy, mine in r["bullets"]],
    )


WARMUP = 25   # rows before a stretch fed to the opponent's feature extractor so that it knows when OUR last shot was (it saturates after 20 ticks)


def enemy_inputs(rows, runs, rects):
    """For every stretch in which the enemy was in sight: [(row index, the network input of the opponent at that row)]."""
    shots = np.cumsum([r["age"] == 0 for r in rows])
    out = []
    for i0, i1 in runs:
        fx = FeatureExtractor(FEATURE_DIM)
        for k in range(max(0, i0 - WARMUP), i0):
            fx(enemy_view(rows, k, False, int(shots[k]), rects))
        for k in range(i0, i1):
            out.append((k, fx(enemy_view(rows, k, k > i0, int(shots[k]), rects))))
    return out


def maybe_blocked(rows, k, rects):
    """The opponent's hull may be touching a wall / the border / our tank at row k: its displacement then says nothing about its throttle
    (the arena keeps the commanded speed while the position stands still)."""
    ex, ey = rows[k]["epos"]
    ox, oy = rows[k]["pos"]
    reach = math.hypot(*HULL_HALF) + 2.0                       # circumscribed circle of the hull + a margin
    if min(ex, ey, W - ex, H - ey) < reach or math.hypot(ex - ox, ey - oy) < 2 * reach:
        return True
    for rx, ry, rw, rh in rects:
        if math.hypot(max(rx - ex, 0.0, ex - (rx + rw)), max(ry - ey, 0.0, ey - (ry + rh))) < reach:
            return True
    return False


def enemy_actions(rows, k, rects=()):
    """The opponent's action in the tick between row k and k+1, reconstructed from our telemetry: (thrust, turn, turret_turn, shot, thrust_known).
    thrust_known is False when the hull may have been blocked (see maybe_blocked)."""
    a, b = rows[k], rows[k + 1]
    ha, hb = a["ebody"], b["ebody"]
    v_a = (a["epos"][0] - rows[k - 1]["epos"][0], a["epos"][1] - rows[k - 1]["epos"][1])
    v_b = (b["epos"][0] - a["epos"][0], b["epos"][1] - a["epos"][1])
    speed_a = v_a[0] * math.cos(ha) + v_a[1] * math.sin(ha)
    speed_b = v_b[0] * math.cos(hb) + v_b[1] * math.sin(hb)
    target = (speed_b - (1.0 - ACCEL) * speed_a) / ACCEL                    # speed += (target - speed) * ACCEL
    thrust = target / (MAX_SPEED if target >= 0 else MAX_REVERSE)
    clip = lambda x: min(max(x, -1.0), 1.0)
    known = not (maybe_blocked(rows, k, rects) or maybe_blocked(rows, k + 1, rects))
    return (clip(thrust), clip(_wrap(hb - ha) / TURN_SPEED), clip(_wrap(b["eturret"] - a["eturret"]) / TURRET_SPEED),
            float(b["age"] == 0), known)


class Adapter:
    def __init__(self, policy, parts="move,shot,bc,rl"):
        parts = set(parts.replace(" ", "").split(",")) if isinstance(parts, str) else set(parts)
        if parts - set(PARTS):
            raise ValueError(f"unknown adaptation parts {sorted(parts - set(PARTS))}; choose from {PARTS}")
        self.pol, self.parts = policy, parts
        self.rows, self.rects = [], None
        self.vel_samples, self.shot_samples = [], []
        self.M0 = np.hstack([policy.p["w_out"], policy.p["b_out"][:, None]]).astype(np.float64)   # the exported output layer (anchor)
        self.history = []

    # ------------------------------------------------------------------ recording (called from act())
    def record(self, raw, fx, h, out, move_raw, shot, sampled):
        if self.rects is None:
            self.rects = [(o[0] * W, o[1] * H, o[2] * W, o[3] * H) for o in raw.get("obstacles", ())]
        own = (raw["x"] * W, raw["y"] * H)
        self.rows.append(dict(
            pos=own, body=raw["body_angle"], turret=raw["turret_angle"], hp=raw["hp"], ammo=raw.get("ammo", 1.0),
            vis=bool(raw["enemy_visible"]), epos=(own[0] + raw["enemy_rel_x"] * W, own[1] + raw["enemy_rel_y"] * H),
            ebody=raw.get("enemy_body_angle", 0.0), eturret=raw["enemy_turret_angle"], ehp=raw["enemy_hp"], dist=raw["enemy_dist"],
            bullets=[(own[0] + b[0] * W, own[1] + b[1] * H, b[2] * BULLET_SPEED, b[3] * BULLET_SPEED, b[4]) for b in raw.get("bullets", ())],
            age=fx.enemy_age, shot_in=fx.shot_input.copy(), lead_err=fx.lead_err,
            h=h.copy(), mv=np.asarray(move_raw, np.float64).copy(), shot=float(shot > 0), sampled=bool(sampled)))

    # ------------------------------------------------------------------ the match that just ended
    @staticmethod
    def _result(rows):
        """+1 we killed the enemy, -1 any failure (killed, timeout), 0 unknown. The last observation is never shown to the agent,
        so a tank at <= 25 HP in the last row is the one that died."""
        if len(rows) >= MAX_STEPS:
            return -1
        mine_low, foe_low = rows[-1]["hp"] <= 25.0, rows[-1]["ehp"] <= 25.0
        return 1 if foe_low and not mine_low else (-1 if mine_low and not foe_low else 0)

    def adapt(self):
        rows, self.rows = self.rows, []
        if len(rows) < MIN_TICKS:
            return None
        result = self._result(rows)
        runs = visible_runs(rows)
        report = dict(ticks=len(rows), result=result, visible_ticks=int(sum(i1 - i0 for i0, i1 in runs)))
        if "move" in self.parts:
            report["move"] = self._fit_move(rows, runs)
        if "shot" in self.parts:
            report["shot"] = self._fit_shot(rows)
        if self.parts & {"bc", "rl"}:
            report["head"] = self._fit_head(rows, runs, result)
        self.history.append(report)
        return report

    # ------------------------------------------------------------------ move: where will it be when the bullet arrives
    @staticmethod
    def _ridge(U, Y):
        """Y ~ A U + b with a ridge prior towards A = I, b = 0. U, Y: (n, 2) in units of MAX_SPEED."""
        X = np.hstack([U, np.ones((len(U), 1))])
        prior = np.hstack([np.eye(2), np.zeros((2, 1))])                       # (2, 3)
        sol = np.linalg.solve(X.T @ X + RIDGE * np.eye(3), X.T @ (Y - X @ prior.T))
        full = prior + sol.T
        return full[:, :2], full[:, 2]

    def _fit_move(self, rows, runs):
        for i0, i1 in runs:
            P = np.array([rows[i]["epos"] for i in range(i0, i1)])
            for k in range(1, len(P) - HORIZON):
                self.vel_samples.append(((P[k] - P[k - 1]) / MAX_SPEED, (P[k + HORIZON] - P[k]) / HORIZON / MAX_SPEED))
        self.vel_samples = self.vel_samples[-SAMPLE_CAP:]
        n = len(self.vel_samples)
        if n < 150:
            return f"only {n} samples"
        U = np.array([s[0] for s in self.vel_samples])
        Y = np.array([s[1] for s in self.vel_samples])
        half = n // 2
        A, b = self._ridge(U[:half], Y[:half])
        err = lambda Ah, bh: float(np.mean(np.sum((Y[half:] - (U[half:] @ Ah.T + bh)) ** 2, axis=1)))
        base, model = err(np.eye(2), np.zeros(2)), err(A, b)
        if base < 1e-5 or model > 0.95 * base:                                 # must beat "keeps its velocity" on data it was not fitted on
            self.pol.fx.vel_model = None
            return f"kept constant velocity (held-out error {model:.4f} vs {base:.4f})"
        A, b = self._ridge(U, Y)
        self.pol.fx.vel_model = (A, b)
        return f"model on (held-out error {model:.4f} vs {base:.4f} for constant velocity)"

    # ------------------------------------------------------------------ shot: when does it fire
    def _fit_shot(self, rows):
        for t in range(len(rows) - 1):
            self.shot_samples.append((rows[t]["shot_in"], float(rows[t + 1]["age"] == 0)))
        self.shot_samples = self.shot_samples[-SAMPLE_CAP:]
        X = np.array([s[0] for s in self.shot_samples])
        y = np.array([s[1] for s in self.shot_samples])
        if y.sum() < 5:
            return f"only {int(y.sum())} shots seen"
        theta0 = np.asarray(SHOT_THETA, np.float64)
        theta = theta0.copy()
        for _ in range(10):                                                    # Newton steps, L2 prior towards the default model
            p = 1.0 / (1.0 + np.exp(-X @ theta))
            grad = X.T @ (p - y) + SHOT_RIDGE * (theta - theta0)
            hess = (X * (p * (1 - p))[:, None]).T @ X + SHOT_RIDGE * np.eye(4)
            theta = theta - np.linalg.solve(hess, grad)
        self.pol.fx.theta = theta
        return f"theta {np.round(theta, 2).tolist()} from {int(y.sum())} shots in {len(y)} ticks"

    # ------------------------------------------------------------------ head: REINFORCE + behaviour cloning on the output layer
    def _rewards(self, rows, result):
        """The training reward as far as telemetry shows it: hits, bad shots, hiding, the outcome (all 'per tick between row t and t+1')."""
        r = np.zeros(len(rows))
        for t in range(len(rows) - 1):
            a, b = rows[t], rows[t + 1]
            r[t] += 15.0 * round((a["ehp"] - b["ehp"]) / 25.0) - 10.0 * round((a["hp"] - b["hp"]) / 25.0)
            if a["ammo"] - b["ammo"] > 0.02:                                   # we fired
                off = abs(b["lead_err"]) > math.atan2(HULL_HALF_WIDTH, b["dist"] * DIAG)
                r[t] -= SHOT_BLIND if not b["vis"] else (SHOT_OFF if off else 0.0)
            r[t] -= 0.02 + (0.01 if not b["vis"] else 0.0)
        r[-1] += 100.0 * result
        return r

    def _advantages(self, rows, result):
        r = self._rewards(rows, result)
        G, g = np.zeros_like(r), 0.0
        for t in reversed(range(len(r))):
            g = r[t] + GAMMA * g
            G[t] = g
        A = G - G.mean()
        s = A.std()
        return A / s if s > 1e-6 else np.zeros_like(A)

    def _bc_data(self, rows, runs, result):
        """(hidden states of OUR network on the opponent's situation, its actions, weights) for the ticks where its action is known."""
        pairs = enemy_inputs(rows, runs, self.rects or [])
        if not pairs:
            return None
        hidden = np.zeros((len(rows), self.pol.hidden))
        have = np.zeros(len(rows), bool)
        pos = 0
        for i0, i1 in runs:                                                    # our network's memory restarts with every stretch
            seq = self.pol.hidden_sequence(np.stack([x for _, x in pairs[pos:pos + i1 - i0]]))
            hidden[i0:i1], have[i0:i1] = seq, True
            pos += i1 - i0
        hurt = np.zeros(len(rows))                                             # ticks shortly before a hit on us
        for t in range(len(rows) - 1):
            if rows[t]["hp"] - rows[t + 1]["hp"] > 1.0:
                hurt[max(0, t - HIT_WINDOW):t + 1] = 1.0
        scale = 0.3 if result > 0 else (1.5 if result < 0 else 1.0)            # copy a winner, not a loser
        Hs, Ts, Ws = [], [], []
        for i0, i1 in runs:
            for k in range(i0 + 1, i1 - 1):                                    # needs the speed at k (so k > i0) and the pose at k + 1
                act = enemy_actions(rows, k, self.rects or ())
                w = scale * (1.0 + 2.0 * hurt[k])
                Hs.append(hidden[k]), Ts.append(act[:4]), Ws.append([w * act[4], w, w, w])    # thrust only where it can be read off the motion
        if len(Hs) < 30:
            return None
        Hb = np.hstack([np.array(Hs), np.ones((len(Hs), 1))])
        T = np.array(Ts)
        return Hb, T[:, :3], T[:, 3], np.array(Ws)

    def _fit_head(self, rows, runs, result):
        pol = self.pol
        T = len(rows)
        Ha = np.hstack([np.stack([r["h"] for r in rows]).astype(np.float64), np.ones((T, 1))])
        use_rl = "rl" in self.parts and rows[0]["sampled"]
        if use_rl:
            A = self._advantages(rows, result)
            MV = np.stack([r["mv"] for r in rows])
            S = np.array([r["shot"] for r in rows])
            sigma = pol.move_noise_scale * np.exp(np.asarray(pol.log_std, np.float64))
        bc = self._bc_data(rows, runs, result) if "bc" in self.parts else None
        if not use_rl and bc is None:
            return "nothing to learn from (no sampled actions / no stretches with the enemy in sight)"
        M = np.hstack([pol.p["w_out"], pol.p["b_out"][:, None]]).astype(np.float64)
        cap = HEAD_MAX_REL * np.linalg.norm(self.M0)
        eta = cap / (2 * HEAD_STEPS)
        unit = lambda G: G / (np.linalg.norm(G) + 1e-12)
        for _ in range(HEAD_STEPS):
            d = -(M - self.M0) / cap                                           # pull back to the exported layer
            if use_rl:
                out = Ha @ M.T
                g = np.empty_like(out)
                g[:, :3] = A[:, None] * (MV - out[:, :3]) / sigma ** 2         # d log pi / d mean of the Gaussian moves
                g[:, 3] = A * (S - _sigmoid(out[:, 3]))                        # ... and of the Bernoulli shot
                d += unit(g.T @ Ha / T)
            if bc is not None:
                Hb, Tm, Ts, Wb = bc                                            # Wb: (n, 4) weight per output (thrust is masked where unknown)
                out = Hb @ M.T
                g = np.empty_like(out)
                g[:, :3] = Tm - out[:, :3]
                g[:, 3] = Ts - _sigmoid(out[:, 3])
                d += unit((g * Wb).T @ Hb / Wb[:, 1].sum())
            M = M + eta * d
        delta = M - self.M0
        if np.linalg.norm(delta) > cap:
            M = self.M0 + delta * cap / np.linalg.norm(delta)
        pol.p["w_out"], pol.p["b_out"] = M[:, :-1].astype(np.float32), M[:, -1].astype(np.float32)
        return (f"head moved {np.linalg.norm(M - self.M0) / np.linalg.norm(self.M0):.3f} of its norm "
                f"(rl {'on' if use_rl else 'off'}, bc {'on, %d ticks' % len(bc[1]) if bc is not None else 'off'})")
