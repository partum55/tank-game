from abc import ABC, abstractmethod
import math
import numpy as np


class ObservationAdapter(ABC):
    """
    Abstract adapter for transforming raw telemetry from TankArena
    into a tensor/vector suitable for a specific neural network architecture.
    """
    @abstractmethod
    def extract(self, raw: dict) -> np.ndarray:
        """Takes raw telemetry dictionary and outputs numpy array."""
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Returns input feature dimension for neural network."""
        pass


def _angle_features(angle: float):
    """Angles go in as (sin, cos): no jump at +-pi."""
    return math.sin(angle), math.cos(angle)


class VectorObservationAdapter(ObservationAdapter):
    """
    27-dimensional normalized feature vector (everything the tank can know):
    - own motion: speed, vx, vy
    - own body angle (sin, cos); turret angle relative to body (sin, cos)
    - own HP, reload timer
    - enemy (live while visible, LAST KNOWN behind a wall): rel_x, rel_y, dist,
      angle to enemy relative to body (sin, cos), enemy turret relative to our body (sin, cos), HP
    - enemy_visible (0/1), enemy_last_seen (0 = just now)
    - 8 lidar rays (walls, obstacles, enemy)
    No absolute x/y on purpose: the map changes every episode, lidar shows the surroundings.
    """
    @property
    def output_dim(self) -> int:
        return 27

    def extract(self, raw: dict) -> np.ndarray:
        body = raw.get("body_angle", 0.0)
        features = [
            raw.get("speed", 0.0),
            raw.get("vx", 0.0),
            raw.get("vy", 0.0),
            *_angle_features(body),
            *_angle_features(raw.get("turret_angle", 0.0) - body),
            raw.get("hp", 100.0) / 100.0,
            raw.get("reload_cooldown", 0.0),
            raw.get("enemy_rel_x", 0.0),
            raw.get("enemy_rel_y", 0.0),
            raw.get("enemy_dist", 0.0),
            *_angle_features(raw.get("enemy_angle_to", 0.0)),
            *_angle_features(raw.get("enemy_turret_angle", 0.0) - body),
            raw.get("enemy_hp", 100.0) / 100.0,
            1.0 if raw.get("enemy_visible", True) else 0.0,
            raw.get("enemy_last_seen", 0.0),
            *raw.get("lidar_rays", [1.0] * 8),
        ]
        return np.array(features, dtype=np.float32)


class LidarObservationAdapter(ObservationAdapter):
    """
    Compact lidar-based vector (num_rays + 7, default 15):
    - lidar distances in [0, 1]
    - speed, hp / 100, reload_cooldown
    - angle to enemy (sin, cos), enemy distance, enemy_visible (0/1)
    """
    def __init__(self, num_rays: int = 8):
        self.num_rays = num_rays

    @property
    def output_dim(self) -> int:
        return self.num_rays + 7

    def extract(self, raw: dict) -> np.ndarray:
        rays = list(raw.get("lidar_rays", [1.0] * self.num_rays))
        rays = (rays + [1.0] * self.num_rays)[: self.num_rays]

        state_features = [
            raw.get("speed", 0.0),
            raw.get("hp", 100.0) / 100.0,
            raw.get("reload_cooldown", 0.0),
            *_angle_features(raw.get("enemy_angle_to", 0.0)),
            raw.get("enemy_dist", 0.0),
            1.0 if raw.get("enemy_visible", True) else 0.0,
        ]
        return np.array(rays + state_features, dtype=np.float32)
