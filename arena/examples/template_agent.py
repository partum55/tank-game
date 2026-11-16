"""
Starting point for YOUR agent. Copy this file, change anything you like, then duel:

    PYTHONPATH=python python3 python/referee.py examples/template_agent.py rule_based --seeds 5 --render

CONTRACT (the only thing the playground cares about):
  - this file defines make_agent()
  - it returns an object with act(obs: dict) -> 4 floats in [-1, 1]:
        [thrust, turn, turret_turn, shoot]   (shoot fires when > 0.5 and the cannon is reloaded)
  - optional reset() is called before every match

Everything else is yours: features, network, framework (NumPy, PyTorch, NEAT, rules...).
Load weights relative to this file (Path(__file__).parent), so the referee can run it from anywhere.

`obs` fields: see docs/CUSTOM_INPUTS_GUIDE.md. Remember the rules:
  - the map is random per seed (and symmetric), do not hard-code coordinates;
  - FOG OF WAR: while `enemy_visible` is False the `enemy_*` fields are the last known values,
    `enemy_last_seen` says how stale they are;
  - bullets are always visible.
"""
import math
from pathlib import Path

import numpy as np

from tank_env import ObservationAdapter, SimpleMLP


class MyAdapter(ObservationAdapter):
    """Example of a custom input: 5 enemy features + 4 cardinal lidar rays + nearest enemy bullet."""

    @property
    def output_dim(self) -> int:
        return 12

    def extract(self, raw: dict) -> np.ndarray:
        lidar = raw.get("lidar_rays", [1.0] * 8)
        # 8 rays relative to the hull, 45 deg apart from -135 to 180: [3] front, [5] +90, [7] back, [1] -90
        front, right, back, left = lidar[3], lidar[5], lidar[7], lidar[1]

        # nearest enemy bullet: [rel_x, rel_y, rel_vx, rel_vy, is_enemy], world axes, normalized
        nearest, bx, by = 1.0, 0.0, 0.0
        for b in raw.get("bullets", []):
            if b[4] == 1.0:
                d = math.hypot(b[0], b[1])
                if d < nearest:
                    nearest, bx, by = d, b[0], b[1]

        angle = raw.get("enemy_angle_to", 0.0)
        return np.array(
            [
                raw.get("enemy_dist", 0.0),
                math.sin(angle),
                math.cos(angle),
                1.0 if raw.get("enemy_visible", True) else 0.0,
                raw.get("hp", 100.0) / 100.0,
                front, right, back, left,
                nearest, bx, by,
            ],
            dtype=np.float32,
        )


class MyAgent:
    def __init__(self):
        self.adapter = MyAdapter()
        self.model = SimpleMLP(self.adapter.output_dim, [32, 16], 4)
        weights = Path(__file__).parent / "my_model.npz"
        if weights.exists():  # train somewhere, save with model.save(...), put it next to this file
            self.model = SimpleMLP.load(str(weights))

    def reset(self):
        pass  # clear recurrent state / memory here if you have any

    def act(self, obs: dict):
        return self.model.forward(self.adapter.extract(obs))


def make_agent():
    return MyAgent()
