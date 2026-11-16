# 🛡️ 2D Tank Duel Playground

A 2D tank combat arena in **Rust** (PyO3 Python bindings, native 60 FPS viewer) built for one thing:
**you and a friend each train your own model, then settle it on the same fair arena.**

The playground is the **arena + the referee**. Your model, features and trainer are yours.

## Why it is a fair fight
- **Random maps, same for both** — every seed builds a different set of walls and spawns (same seed ⇒ identical map). Layouts are point-symmetric, tanks always fit through every gap.
- **Fog of war** — behind a wall the enemy info is the last known snapshot (`enemy_visible`, `enemy_last_seen`).
- **Sides swapped** — the referee plays every map twice, once per side.
- **Sanitized actions** — NaN/inf/out-of-range outputs can't reach the physics.
- **One laptop, two sandboxes** — each agent file runs in its own process (own venv via `--python-a/--python-b`), with a per-move time limit; a crash or hang forfeits only that match.
- **Fast** — > 100k env steps/s headless (rule-based bots, measured on one core).

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin pytest numpy
maturin develop --release
```

```bash
# 1. train the example GA agent  ->  checkpoints/agent.py
PYTHONPATH=python python3 python/train_ga.py --generations 30 --population 30 --seed 1

# 2. duel any two agents (headless scoreboard, 20 maps x 2 sides)
PYTHONPATH=python python3 python/referee.py checkpoints/agent.py rule_based --seeds 20

# 3. watch a duel
PYTHONPATH=python python3 python/referee.py checkpoints/agent.py rule_based --seeds 2 --render -v

# window controls: PAUSE / QUIT buttons in the top strip (also Space = pause, Esc = quit)

# native viewer without Python
cargo run --release --bin standalone_viewer
```

## Your agent = one `.py` file

```python
def make_agent():
    return MyAgent()   # act(obs: dict) -> [thrust, turn, turret_turn, shoot], optional reset()
```

Copy [`examples/template_agent.py`](examples/template_agent.py) and make it yours (PyTorch, NEAT, rules — anything).
Contract, rules and fair-final protocol: [`docs/COLLABORATION_GUIDE.md`](docs/COLLABORATION_GUIDE.md).
Observation fields and fog details: [`docs/CUSTOM_INPUTS_GUIDE.md`](docs/CUSTOM_INPUTS_GUIDE.md).

The GA trainer (`python/train_ga.py`, `SimpleMLP`, adapters, bots) is just an **example** — write your own
trainer against `TankEnvironment` (`reset(seed)`, `step(a1, a2)`).

## 📂 Layout

```
crates/tank_core/     Rust engine: physics, map generation (mapgen.rs), fog of war, renderer
crates/tank_pyo3/     PyO3 bindings (module `tank_arena`)
python/tank_env/      Python SDK: environment, agent loader, adapters, bots, SimpleMLP
python/referee.py     official duel runner
python/train_ga.py    example genetic-algorithm trainer
examples/             template_agent.py
docs/                 collaboration guide, telemetry reference
tests/                Python tests (+ Rust tests in crates/tank_core/tests)
```

## 🧪 Tests

```bash
cargo test -p tank_core
PYTHONPATH=python pytest tests/
```

## Rules version
`tank_env.RULES_VERSION` — if physics, map generation or the contract change it must be bumped; both players
must duel on the same version (tag `rules-v1.2`: 20 shots per match, rectangular 40 × 26 tanks).
