# 📊 Telemetry & Custom Inputs

The Rust engine hands every agent a raw telemetry dict each tick (`obs` in `act(obs)`).
Turn it into whatever input your model wants — a custom `ObservationAdapter` or just your own code
(see `examples/template_agent.py`).

---

## 🌫️ Fog of war (important)

The enemy is **visible** when the straight line between the two tank centres crosses no obstacle.

- visible → every `enemy_*` field is live.
- hidden → `enemy_rel_x/y`, `enemy_dist`, `enemy_angle_to`, `enemy_turret_angle`, `enemy_body_angle`, `enemy_hp`
  hold the **last known** values (updated every tick while visible); `enemy_last_seen`
  tells how stale they are. Your own position keeps changing, so the relative fields keep moving too.
- at the start of an episode both tanks know where the enemy is.
- **bullets are always visible**, lidar is physical (it hits the nearest wall/obstacle/enemy).

## 📡 Raw telemetry schema

| Key | Type | Description |
|---|---|---|
| `x`, `y` | `float` in `[0, 1]` | Own position (`pos / arena size`) |
| `vx`, `vy` | `float` | Own velocity (`/ max speed`) |
| `speed` | `float` | Forward (>0) / reverse (<0) speed (`/ max speed`) |
| `body_angle` | `float` `[-π, π]` | Hull angle in radians (world frame) |
| `turret_angle` | `float` `[-π, π]` | Turret angle in radians (world frame) |
| `hp` | `float` `[0, 100]` | Own HP |
| `reload_cooldown` | `float` `[0, 1]` | Reload timer left (0.0 = ready to fire) |
| `ammo` | `float` `[0, 1]` | Shots left `/ 20` (rules v1.2: 20 per match, no resupply) |
| `enemy_rel_x`, `enemy_rel_y` | `float` | Enemy position relative to you (`/ arena size`), **fog applies** |
| `enemy_dist` | `float` `[0, 1]` | Distance to enemy (`/ arena diagonal`), **fog applies** |
| `enemy_angle_to` | `float` `[-π, π]` | Direction to the enemy relative to your hull heading, **fog applies** |
| `enemy_turret_angle` | `float` `[-π, π]` | Enemy turret angle (world frame), **fog applies** |
| `enemy_body_angle` | `float` `[-π, π]` | Enemy hull angle (world frame; the hull is a 40 × 26 rectangle), **fog applies** |
| `enemy_hp` | `float` `[0, 100]` | Enemy HP, **fog applies** |
| `enemy_visible` | `bool` | Direct line of sight to the enemy right now |
| `enemy_last_seen` | `float` `[0, 1]` | Ticks since the enemy was last seen `/ max_steps` (0 while visible) |
| `lidar_rays` | `list[float]` ×8 | Distance to nearest wall/obstacle/enemy at hull-relative angles −135°, −90°, −45°, 0°, 45°, 90°, 135°, 180°, `/ arena diagonal` |
| `obstacles` | `list[[x, y, w, h]]` | The whole map this episode, normalized by arena width/height (top-left corner + size) |
| `bullets` | `list[[rel_x, rel_y, rel_vx, rel_vy, is_enemy]]` | All active bullets, relative to you |

Tips:
- Angles wrap at ±π — feed `sin`/`cos` to a network instead of the raw value.
- The map changes every seed. Absolute `x, y` invite memorizing; `obstacles` + lidar generalize.
- Maps are point-symmetric (180° rotation about the centre); you may be tank 1 **or** tank 2.

## 🛠 Custom adapter

```python
from tank_env import ObservationAdapter
import math, numpy as np

class PolarAdapter(ObservationAdapter):
    @property
    def output_dim(self) -> int:
        return 6

    def extract(self, raw: dict) -> np.ndarray:
        a = raw["enemy_angle_to"]
        return np.array([
            raw["enemy_dist"], math.sin(a), math.cos(a),
            raw["hp"] / 100.0, raw["reload_cooldown"], float(raw["enemy_visible"]),
        ], dtype=np.float32)
```

Built-ins: `VectorObservationAdapter` (27 dims, full state + lidar) and `LidarObservationAdapter` (15 dims).

Full example with bullet sensing: [examples/template_agent.py](../examples/template_agent.py).
