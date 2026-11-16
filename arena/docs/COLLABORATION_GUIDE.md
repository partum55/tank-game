# 🤝 Collaboration Guide: duel your models

You and your friend each train your own model, in your own way, and settle it on this shared
playground. The playground is the **arena + the referee**; models are yours.

---

## 1. Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin pytest numpy
maturin develop --release
```

**Both of you must use the same rules.** Check `tank_env.RULES_VERSION` (now **1.2**; git tag `rules-v1.2`).
If the physics, map generation or contract changes, bump it and both of you pull before the next duel.

## 2. The contract: an agent is a `.py` file

```python
def make_agent():
    return MyAgent()          # any object with  act(obs) -> [thrust, turn, turret_turn, shoot]
```

| Index | Action | Range | Meaning |
|---|---|---|---|
| `[0]` | `thrust` | `[-1, 1]` | forward (>0) / reverse (<0) |
| `[1]` | `turn` | `[-1, 1]` | hull steering left (<0) / right (>0) |
| `[2]` | `turret_turn` | `[-1, 1]` | turret left (<0) / right (>0) |
| `[3]` | `shoot` | `[-1, 1]` | fire if `> 0.5` and reloaded |

Inputs, network, framework, preprocessing are entirely yours (one can use PyTorch + PPO, the other a
GA with `SimpleMLP`). Weights live next to the file and are loaded via `Path(__file__).parent`.
Start from [`examples/template_agent.py`](../examples/template_agent.py). Observation fields:
[`CUSTOM_INPUTS_GUIDE.md`](CUSTOM_INPUTS_GUIDE.md). NaN/out-of-range outputs are sanitized by the engine.

## 3. Rules of the arena

- **Random maps**: every seed gives a different set of walls and spawns (same seed ⇒ identical map).
  Maps are point-symmetric, so both sides are equal; wall gaps always fit a tank.
- **Fog of war**: behind a wall the enemy info is the *last known* snapshot (`enemy_visible`, `enemy_last_seen`).
- Bullets stop at walls; a tank pressed against a wall cannot shoot through it (the shot is still spent).
- **Ammo (v1.2)**: every tank has **20 shots per match**, no resupply (reload between shots stays 20 ticks). With 0 left `shoot` does nothing.
  Own ammo is in the telemetry (`ammo`, 1.0 = full); the enemy's is not, but every bullet it fires is visible.
- **Rectangular tanks (v1.2)**: the hull is a 40 × 26 px rectangle that turns with `body_angle`; bullets, walls, the border, the other tank and
  the lidar all see this rectangle (not a circle). `enemy_body_angle` tells how the enemy hull is turned.
- Match ends at 0 HP or after 1500 ticks (more HP wins; equal HP = draw).
- Agents live for the whole series: the referee calls `reset()` before each match but keeps the process, so an agent may remember
  the opponent between matches (`reset()` has up to 5 s; keep it deterministic).
- The referee plays **every seed twice with sides swapped**.

## 4. Train

```bash
PYTHONPATH=python python3 python/train_ga.py --generations 30 --population 30 --seed 1
# -> checkpoints/best_model.npz + checkpoints/agent.py
```

Opponents (`--opponent`): `rule_based`, `random`, **any agent file** (e.g. your friend's, or your previous
best) or `self` (best of the previous generation). Every generation uses fresh maps, so the model cannot
memorize one scene. Training is the example GA — write your own trainer if you prefer;
`TankEnvironment` is a plain gym-like API (`reset(seed)`, `step(a1, a2)`).

## 5. Duel (both models on one laptop)

Each agent file runs **in its own process** (`python/agent_host.py`, started by the referee). So:
- you can use **different venvs** (yours: numpy, friend's: PyTorch) — point the referee at each interpreter;
- helper modules with the same name (`utils.py`, `model.py`) do not clash;
- an agent can't reach into the arena or the other agent;
- a crash / hang / `act()` slower than `--act-timeout` (default 500 ms) = **forfeit of that match** (counted as a loss, the duel continues).

```bash
# headless, 20 maps x 2 sides, each agent in its own venv:
PYTHONPATH=python python3 python/referee.py my_agent.py friend_agent.py --seeds 20 \
    --python-a .venv/bin/python --python-b ~/friend/.venv/bin/python

# watch it:
PYTHONPATH=python python3 python/referee.py my_agent.py friend_agent.py --seeds 2 --render -v
```

The agent's venv only needs what *your agent* imports (NumPy, PyTorch…); the Rust extension is **not** needed there
(the referee process owns the arena). The referee runs the agents one after another each tick, so they never compete for CPU
inside a match; the weights must load in the agent's `make_agent()` (loading time isn't counted, up to 2 min).

Tips for a laptop: `torch.set_num_threads(1)` and a small network keep `act()` in microseconds; a GPU is not worth it for
this size (host↔device copies cost more than the forward pass). `--no-isolate` runs agent files inside the referee process for debugging only.

Output: W/L/D/F (forfeits) per agent, average final HP and `act()` time (includes ~0.1 ms of process messaging).

### Fair final
1. Agree on the rules version and the number of maps (≥ 50).
2. Practice with `--seed-base 1000` (default). **Do not train on the final seeds.**
3. One of you picks a secret `--seed-base` for the final and reveals it only when both agents are frozen.
4. Both run the *same* command on one machine. The arena is deterministic for a given seed; the result is
   reproducible as long as both agents are deterministic (seed any randomness inside your agent, e.g. in `reset()`).
