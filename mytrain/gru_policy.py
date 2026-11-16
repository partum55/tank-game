"""
Recurrent (GRU) policy in plain NumPy - STANDALONE (needs only numpy).

Used by the training workers (hall-of-fame opponents), by evaluation, and copied next to every exported
agent, so the duel venv needs neither PyTorch nor this package.

Network:  obs -> tanh(Linear) -> GRU -> Linear -> [mean_thrust, mean_turn, mean_turret, shoot_logit]
Weight layout is PyTorch's (`nn.Linear`: (out, in); `nn.GRU`: gate order r, z, n).
"""
import math
import os
import shutil

import numpy as np

PARAM_KEYS = ("w_in", "b_in", "w_ih", "w_hh", "b_ih", "b_hh", "w_out", "b_out")
OPTIONAL_KEYS = ("log_std",)  # exploration noise of the moves; when present the agent acts like it was trained (see below)
BASE_DIM = 27       # position-free state, enemy (live or last known), fog flags, lidar
VEL_DIM = 31        # + enemy velocity (2) and the aim error to the LEAD point (sin, cos)
FEATURE_DIM = 36    # + own ammo, nearest incoming enemy bullet (urgency, miss distance, side), probability that the enemy is shooting
W, H, MAX_SPEED, BULLET_SPEED = 1200.0, 800.0, 4.5, 9.0
RELOAD_TICKS = 20
THREAT_TICKS = 60.0       # an enemy bullet counts as a threat if its closest approach to us is within this many ticks
THREAT_SCALE = 40.0       # px: miss distance / lateral offset are divided by this (a hull is ~26-40 px)
NO_SHOT = 1000            # "enemy has not fired yet"
# start values of the enemy-shot model p = sigmoid(theta . [1, |enemy turret error towards us| / pi, ticks since its last shot / 20 (<=1), distance]);
# a duel agent may refit them on the opponent it has just played (see adapt.py)
SHOT_THETA = (-1.5, -8.0, 3.0, -2.0)
MOVE_NOISE_SCALE = 0.5   # moves are played with exp(log_std) * 0.5: measured better than the full training noise (0.91 vs 0.83) and calmer
STANDALONE_FILE = "tank_gru_policy.py"
ADAPT_FILE = "tank_adapt.py"
AGENT_FILE = '''"""Auto-generated - NumPy only. Duel it: python referee.py <this file> <other agent>"""
from pathlib import Path
from tank_gru_policy import NumpyGRUPolicy


def make_agent():
    return NumpyGRUPolicy.load(Path(__file__).parent / "policy.npz"__ARGS__)
'''


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def base_features(raw: dict) -> np.ndarray:
    """27 normalized features from the raw telemetry (angles as sin/cos, no absolute x/y)."""
    body = raw.get("body_angle", 0.0)

    def sc(a):
        return math.sin(a), math.cos(a)

    f = [
        raw.get("speed", 0.0), raw.get("vx", 0.0), raw.get("vy", 0.0),
        *sc(body),
        *sc(raw.get("turret_angle", 0.0) - body),
        raw.get("hp", 100.0) / 100.0,
        raw.get("reload_cooldown", 0.0),
        raw.get("enemy_rel_x", 0.0), raw.get("enemy_rel_y", 0.0), raw.get("enemy_dist", 0.0),
        *sc(raw.get("enemy_angle_to", 0.0)),
        *sc(raw.get("enemy_turret_angle", 0.0) - body),
        raw.get("enemy_hp", 100.0) / 100.0,
        1.0 if raw.get("enemy_visible", True) else 0.0,
        raw.get("enemy_last_seen", 0.0),
        *raw.get("lidar_rays", [1.0] * 8),
    ]
    return np.array(f, dtype=np.float32)


def lead_angle_error(px, py, vx, vy, turret):
    """Where to shoot at a moving target: angle between the turret and the intercept point (bullet speed is constant).
    p = enemy - own (px), v = enemy velocity (px/tick). |p + v t| = s t has exactly one positive root since |v| < s."""
    s2 = BULLET_SPEED * BULLET_SPEED
    a = min(vx * vx + vy * vy - s2, -1e-6)
    b = 2.0 * (px * vx + py * vy)
    c = px * px + py * py
    t = (-b - math.sqrt(max(b * b - 4.0 * a * c, 0.0))) / (2.0 * a)
    t = min(max(t, 0.0), 400.0)
    err = math.atan2(py + vy * t, px + vx * t) - turret
    return math.atan2(math.sin(err), math.cos(err))


def bullet_threat(raw: dict, own, body: float):
    """Nearest-in-time incoming enemy bullet: (urgency 1 = imminent, miss distance / 40 px (2 = none), side = signed lateral offset of the
    closest approach in the own-heading frame / 40 px). Only bullets that approach and come closest within THREAT_TICKS count."""
    best = None
    for rx, ry, vx, vy, is_enemy in raw.get("bullets", ()):
        if not is_enemy:
            continue
        px, py, vx, vy = rx * W, ry * H, vx * BULLET_SPEED, vy * BULLET_SPEED
        pv, vv = px * vx + py * vy, max(vx * vx + vy * vy, 1e-6)
        tc = -pv / vv
        if pv >= 0 or tc > THREAT_TICKS or (best is not None and tc >= best[0]):
            continue
        best = (tc, px + vx * tc, py + vy * tc)
    if best is None:
        return 0.0, 2.0, 0.0
    tc, mx, my = best
    lateral = -math.sin(body) * mx + math.cos(body) * my
    return 1.0 - tc / THREAT_TICKS, min(math.hypot(mx, my) / THREAT_SCALE, 2.0), min(max(lateral / THREAT_SCALE, -1.0), 1.0)


def enemy_shot_input(turret_err: float, age: int, dist: float):
    """The model-free inputs of the enemy-shot probability: [1, |turret error| / pi, reload readiness (0..1), distance]."""
    return np.array([1.0, abs(turret_err) / math.pi, min(age, RELOAD_TICKS) / RELOAD_TICKS, dist])


class FeatureExtractor:
    """
    The model's inputs, computed from the raw telemetry. STATEFUL: the enemy's velocity is the change of its position between two
    consecutive ticks in which it was visible (0 otherwise), and its last shot is a new enemy bullet that no bullet of the previous tick
    explains - so call it once per tick and `reset()` between matches.
    dim 27 = the original inputs (older exports), dim 31 = + enemy velocity (own-heading frame) and the lead-aim error,
    dim 36 = + ammo, incoming-bullet threat (3) and the enemy-shot probability (rules v1.2).
    """

    def __init__(self, dim: int = FEATURE_DIM, theta=None):
        if dim not in (BASE_DIM, VEL_DIM, FEATURE_DIM):
            raise ValueError(f"unsupported feature size {dim} (expected {BASE_DIM}, {VEL_DIM} or {FEATURE_DIM})")
        self.dim = dim
        self.theta = np.asarray(SHOT_THETA if theta is None else theta, np.float64)
        # optional learned model of the enemy's motion (see adapt.py): predicted velocity = A @ v + b, both in units of MAX_SPEED.
        # None = the enemy keeps its measured velocity (what the network was trained with).
        self.vel_model = None
        self.reset()

    def reset(self):
        self.prev = None          # (enemy position px, was visible) of the previous tick
        self.lead_err = 0.0
        self.enemy_vel = (0.0, 0.0)   # measured enemy velocity, world px/tick (0 while unknown)
        self.prev_bullets = []    # enemy bullets of the previous tick in world coordinates (x, y, vx, vy)
        self.enemy_age = NO_SHOT  # ticks since the enemy last fired
        self.shot_input = enemy_shot_input(math.pi, NO_SHOT, 1.0)   # inputs of the shot model on the last call (for refitting theta)

    def _enemy_age(self, raw: dict, own):
        cur = [(own[0] + rx * W, own[1] + ry * H, vx * BULLET_SPEED, vy * BULLET_SPEED)
               for rx, ry, vx, vy, is_enemy in raw.get("bullets", ()) if is_enemy]
        fresh = any(all(abs(x - (px + pvx)) > 0.1 or abs(y - (py + pvy)) > 0.1 for px, py, pvx, pvy in self.prev_bullets)
                    for x, y, _, _ in cur)
        self.enemy_age = 0 if fresh else min(self.enemy_age + 1, NO_SHOT)
        self.prev_bullets = cur

    def __call__(self, raw: dict) -> np.ndarray:
        base = base_features(raw)
        if self.dim == BASE_DIM:
            return base
        own = (raw.get("x", 0.0) * W, raw.get("y", 0.0) * H)
        seen = (own[0] + raw.get("enemy_rel_x", 0.0) * W, own[1] + raw.get("enemy_rel_y", 0.0) * H)   # live, or last known
        visible = bool(raw.get("enemy_visible", True))
        vx = vy = 0.0
        if self.prev is not None and visible and self.prev[1]:
            vx, vy = seen[0] - self.prev[0][0], seen[1] - self.prev[0][1]
        self.prev = (seen, visible)
        body = raw.get("body_angle", 0.0)
        c, s = math.cos(body), math.sin(body)
        v_fwd, v_side = (c * vx + s * vy) / MAX_SPEED, (-s * vx + c * vy) / MAX_SPEED
        self.enemy_vel = (vx, vy)
        if self.vel_model is not None:
            pred = self.vel_model[0] @ np.array([vx, vy]) / MAX_SPEED + self.vel_model[1]
            vx, vy = float(pred[0]) * MAX_SPEED, float(pred[1]) * MAX_SPEED
        self.lead_err = lead_angle_error(seen[0] - own[0], seen[1] - own[1], vx, vy, raw.get("turret_angle", 0.0))
        feats = np.concatenate([base, np.array([min(max(v_fwd, -2.0), 2.0), min(max(v_side, -2.0), 2.0),
                                                math.sin(self.lead_err), math.cos(self.lead_err)], np.float32)])
        if self.dim == VEL_DIM:
            return feats
        urgency, miss, side = bullet_threat(raw, own, body)
        self._enemy_age(raw, own)
        d = raw.get("enemy_turret_angle", 0.0) - math.atan2(own[1] - seen[1], own[0] - seen[0])   # enemy turret vs the direction enemy -> us
        turret_err = math.atan2(math.sin(d), math.cos(d))
        self.shot_input = enemy_shot_input(turret_err, self.enemy_age, raw.get("enemy_dist", 0.0))
        shot_p = 1.0 / (1.0 + math.exp(-float(self.theta @ self.shot_input)))
        return np.concatenate([feats, np.array([raw.get("ammo", 1.0), urgency, miss, side, shot_p], np.float32)])


def random_params(in_dim: int = FEATURE_DIM, emb: int = 64, hidden: int = 64, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)

    def lin(o, i):
        k = 1.0 / np.sqrt(i)
        return rng.uniform(-k, k, (o, i)).astype(np.float32), rng.uniform(-k, k, o).astype(np.float32)

    w_in, b_in = lin(emb, in_dim)
    w_ih, b_ih = lin(3 * hidden, emb)
    w_hh, b_hh = lin(3 * hidden, hidden)
    w_out, b_out = lin(4, hidden)
    return dict(w_in=w_in, b_in=b_in, w_ih=w_ih, w_hh=w_hh, b_ih=b_ih, b_hh=b_hh, w_out=w_out, b_out=b_out)


def gru_step(p: dict, h: np.ndarray, x: np.ndarray) -> np.ndarray:
    """One step of obs -> tanh(Linear) -> GRU: the new hidden state (the head `w_out @ h + b_out` is applied by the caller)."""
    H = h.shape[0]
    e = np.tanh(p["w_in"] @ x + p["b_in"])
    gi = p["w_ih"] @ e + p["b_ih"]
    gh = p["w_hh"] @ h + p["b_hh"]
    r = _sigmoid(gi[:H] + gh[:H])
    z = _sigmoid(gi[H:2 * H] + gh[H:2 * H])
    n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
    return ((1.0 - z) * n + z * h).astype(np.float32)


class NumpyGRUPolicy:
    """
    Deterministic agent: act(obs) -> [thrust, turn, turret_turn, shoot]; reset() clears memory.
    `adapt` (a subset of "move,shot,bc,rl", see adapt.py): before every match after the first, reset() learns from the match that just
    ended (the opponent's motion, its shots, its successful actions, our own outcome); None = a plain frozen agent.
    """

    def __init__(self, params: dict, deterministic: bool = False, seed: int = 12345, move_noise_scale=None, adapt=None):
        self.p = {k: np.asarray(params[k], dtype=np.float32) for k in PARAM_KEYS}
        # PPO trains a STOCHASTIC policy (Gaussian moves, Bernoulli shot). Such a policy can depend on its own noise, and the
        # mean/threshold version of it may play far worse (measured: 0.89 vs 0.04 against the bot). So when the exploration
        # noise is stored with the weights the agent keeps sampling, with a fixed seed re-applied on every reset().
        self.log_std = np.asarray(params["log_std"], dtype=np.float32) if "log_std" in params else None
        self.sample = self.log_std is not None and not deterministic
        self._seed = seed
        self.move_noise_scale = MOVE_NOISE_SCALE if move_noise_scale is None else move_noise_scale
        self.rng = np.random.default_rng(seed)
        self.hidden = self.p["w_hh"].shape[1]
        self.in_dim = self.p["w_in"].shape[1]
        self.fx = FeatureExtractor(self.in_dim)   # raises for sizes this code does not know
        self.h = np.zeros(self.hidden, dtype=np.float32)
        self.adapter = None
        if adapt:
            if self.in_dim != FEATURE_DIM:
                raise ValueError(f"adaptation needs the {FEATURE_DIM}-input policy, this one has {self.in_dim}")
            try:
                from .adapt import Adapter
            except ImportError:      # standalone export: tank_adapt.py sits next to tank_gru_policy.py
                from tank_adapt import Adapter
            self.adapter = Adapter(self, adapt)

    def reset(self):
        if self.adapter is not None:
            self.adapter.adapt()              # learn from the match that just ended (no-op before the first match)
        self.h = np.zeros(self.hidden, dtype=np.float32)
        self.fx.reset()
        self.rng = np.random.default_rng(self._seed)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.h = gru_step(self.p, self.h, x)
        return self.p["w_out"] @ self.h + self.p["b_out"]

    def hidden_sequence(self, xs: np.ndarray) -> np.ndarray:
        """Hidden states of this network over the input sequence xs (T, in_dim), starting from a zero memory; leaves the agent's own state alone."""
        h, out = np.zeros(self.hidden, dtype=np.float32), []
        for x in xs:
            h = gru_step(self.p, h, x)
            out.append(h)
        return np.stack(out) if out else np.zeros((0, self.hidden), np.float32)

    def act(self, obs: dict):
        out = self.forward(self.fx(obs))
        if self.sample:
            move_raw = out[:3] + self.move_noise_scale * np.exp(self.log_std) * self.rng.standard_normal(3)
            move = np.clip(move_raw, -1.0, 1.0)
            shoot = 1.0 if self.rng.random() < _sigmoid(out[3]) else -1.0
        else:
            move_raw = out[:3]
            move = np.clip(move_raw, -1.0, 1.0)
            shoot = 1.0 if out[3] > 0.0 else -1.0
        if self.adapter is not None:
            self.adapter.record(obs, self.fx, self.h, out, move_raw, shoot, self.sample)
        return [float(move[0]), float(move[1]), float(move[2]), shoot]

    def save(self, path: str):
        extra = {} if self.log_std is None else {"log_std": self.log_std}
        np.savez_compressed(path, **self.p, **extra)

    @classmethod
    def load(cls, path: str, deterministic: bool = False, move_noise_scale=None, adapt=None) -> "NumpyGRUPolicy":
        data = np.load(path)
        return cls({k: data[k] for k in PARAM_KEYS + tuple(k for k in OPTIONAL_KEYS if k in data.files)}, deterministic,
                   move_noise_scale=move_noise_scale, adapt=adapt)


def write_standalone_agent(folder: str, params: dict, move_noise_scale=None, adapt=None):
    """folder/{agent.py, policy.npz, tank_gru_policy.py, tank_adapt.py}: a self-contained agent for referee.py.
    `adapt` = e.g. "move,shot" switches on the pre-match adaptation (adapt.py) in the exported agent."""
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(os.path.join(folder, "policy.npz"),
                        **{k: params[k] for k in PARAM_KEYS + OPTIONAL_KEYS if k in params})
    shutil.copyfile(__file__, os.path.join(folder, STANDALONE_FILE))
    shutil.copyfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapt.py"), os.path.join(folder, ADAPT_FILE))
    args = "" if move_noise_scale is None else f", move_noise_scale={move_noise_scale}"
    args += "" if not adapt else f", adapt={adapt!r}"
    with open(os.path.join(folder, "agent.py"), "w") as f:
        f.write(AGENT_FILE.replace("__ARGS__", args))
