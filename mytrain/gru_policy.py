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
FEATURE_DIM = 31    # + enemy velocity (2) and the aim error to the LEAD point (sin, cos)
W, H, MAX_SPEED, BULLET_SPEED = 1200.0, 800.0, 4.5, 9.0
MOVE_NOISE_SCALE = 0.5   # moves are played with exp(log_std) * 0.5: measured better than the full training noise (0.91 vs 0.83) and calmer
STANDALONE_FILE = "tank_gru_policy.py"
AGENT_FILE = '''"""Auto-generated - NumPy only. Duel it: python referee.py <this file> <other agent>"""
from pathlib import Path
from tank_gru_policy import NumpyGRUPolicy


def make_agent():
    return NumpyGRUPolicy.load(Path(__file__).parent / "policy.npz")
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


class FeatureExtractor:
    """
    The model's inputs, computed from the raw telemetry. STATEFUL: the enemy's velocity is the change of its position between two
    consecutive ticks in which it was visible (0 otherwise), so call it once per tick and `reset()` between matches.
    dim 27 = the original inputs (older exports), dim 31 = + enemy velocity (own-heading frame) and the lead-aim error.
    """

    def __init__(self, dim: int = FEATURE_DIM):
        if dim not in (BASE_DIM, FEATURE_DIM):
            raise ValueError(f"unsupported feature size {dim} (expected {BASE_DIM} or {FEATURE_DIM})")
        self.dim = dim
        self.reset()

    def reset(self):
        self.prev = None          # (enemy position px, was visible) of the previous tick
        self.lead_err = 0.0

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
        self.lead_err = lead_angle_error(seen[0] - own[0], seen[1] - own[1], vx, vy, raw.get("turret_angle", 0.0))
        return np.concatenate([base, np.array([min(max(v_fwd, -2.0), 2.0), min(max(v_side, -2.0), 2.0),
                                               math.sin(self.lead_err), math.cos(self.lead_err)], np.float32)])


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


class NumpyGRUPolicy:
    """Deterministic agent: act(obs) -> [thrust, turn, turret_turn, shoot]; reset() clears memory."""

    def __init__(self, params: dict, deterministic: bool = False, seed: int = 12345):
        self.p = {k: np.asarray(params[k], dtype=np.float32) for k in PARAM_KEYS}
        # PPO trains a STOCHASTIC policy (Gaussian moves, Bernoulli shot). Such a policy can depend on its own noise, and the
        # mean/threshold version of it may play far worse (measured: 0.89 vs 0.04 against the bot). So when the exploration
        # noise is stored with the weights the agent keeps sampling, with a fixed seed re-applied on every reset().
        self.log_std = np.asarray(params["log_std"], dtype=np.float32) if "log_std" in params else None
        self.sample = self.log_std is not None and not deterministic
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self.hidden = self.p["w_hh"].shape[1]
        self.in_dim = self.p["w_in"].shape[1]
        self.fx = FeatureExtractor(self.in_dim)   # raises for sizes this code does not know
        self.h = np.zeros(self.hidden, dtype=np.float32)

    def reset(self):
        self.h = np.zeros(self.hidden, dtype=np.float32)
        self.fx.reset()
        self.rng = np.random.default_rng(self._seed)

    def forward(self, x: np.ndarray) -> np.ndarray:
        p, H = self.p, self.hidden
        e = np.tanh(p["w_in"] @ x + p["b_in"])
        gi = p["w_ih"] @ e + p["b_ih"]
        gh = p["w_hh"] @ self.h + p["b_hh"]
        r = _sigmoid(gi[:H] + gh[:H])
        z = _sigmoid(gi[H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        self.h = ((1.0 - z) * n + z * self.h).astype(np.float32)
        return p["w_out"] @ self.h + p["b_out"]

    def act(self, obs: dict):
        out = self.forward(self.fx(obs))
        if self.sample:
            move = np.clip(out[:3] + MOVE_NOISE_SCALE * np.exp(self.log_std) * self.rng.standard_normal(3), -1.0, 1.0)
            shoot = 1.0 if self.rng.random() < _sigmoid(out[3]) else -1.0
        else:
            move = np.clip(out[:3], -1.0, 1.0)
            shoot = 1.0 if out[3] > 0.0 else -1.0
        return [float(move[0]), float(move[1]), float(move[2]), shoot]

    def save(self, path: str):
        extra = {} if self.log_std is None else {"log_std": self.log_std}
        np.savez_compressed(path, **self.p, **extra)

    @classmethod
    def load(cls, path: str, deterministic: bool = False) -> "NumpyGRUPolicy":
        data = np.load(path)
        return cls({k: data[k] for k in PARAM_KEYS + tuple(k for k in OPTIONAL_KEYS if k in data.files)}, deterministic)


def write_standalone_agent(folder: str, params: dict):
    """folder/{agent.py, policy.npz, tank_gru_policy.py}: a self-contained agent for referee.py."""
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(os.path.join(folder, "policy.npz"),
                        **{k: params[k] for k in PARAM_KEYS + OPTIONAL_KEYS if k in params})
    shutil.copyfile(__file__, os.path.join(folder, STANDALONE_FILE))
    with open(os.path.join(folder, "agent.py"), "w") as f:
        f.write(AGENT_FILE)
