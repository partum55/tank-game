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
FEATURE_DIM = 27
STANDALONE_FILE = "tank_gru_policy.py"
AGENT_FILE = '''"""Auto-generated - NumPy only. Duel it: python referee.py <this file> <other agent>"""
from pathlib import Path
from tank_gru_policy import NumpyGRUPolicy


def make_agent():
    return NumpyGRUPolicy.load(Path(__file__).parent / "policy.npz")
'''


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def features(raw: dict) -> np.ndarray:
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

    def __init__(self, params: dict):
        self.p = {k: np.asarray(params[k], dtype=np.float32) for k in PARAM_KEYS}
        self.hidden = self.p["w_hh"].shape[1]
        self.in_dim = self.p["w_in"].shape[1]
        if self.in_dim != FEATURE_DIM:
            raise ValueError(f"policy expects {self.in_dim} inputs, features() gives {FEATURE_DIM}")
        self.h = np.zeros(self.hidden, dtype=np.float32)

    def reset(self):
        self.h = np.zeros(self.hidden, dtype=np.float32)

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
        out = self.forward(features(obs))
        move = np.clip(out[:3], -1.0, 1.0)
        return [float(move[0]), float(move[1]), float(move[2]), 1.0 if out[3] > 0.0 else -1.0]

    def save(self, path: str):
        np.savez_compressed(path, **self.p)

    @classmethod
    def load(cls, path: str) -> "NumpyGRUPolicy":
        data = np.load(path)
        return cls({k: data[k] for k in PARAM_KEYS})


def write_standalone_agent(folder: str, params: dict):
    """folder/{agent.py, policy.npz, tank_gru_policy.py}: a self-contained agent for referee.py."""
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(os.path.join(folder, "policy.npz"), **{k: params[k] for k in PARAM_KEYS})
    shutil.copyfile(__file__, os.path.join(folder, STANDALONE_FILE))
    with open(os.path.join(folder, "agent.py"), "w") as f:
        f.write(AGENT_FILE)
