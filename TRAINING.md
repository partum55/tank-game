# 🧠 Training: PPO + GRU + hall-of-fame self-play

Train on Colab (A100 for the network, all vCPUs for the arenas), duel on the laptop.
Start from [`colab/train_colab.ipynb`](colab/train_colab.ipynb) — it builds the arena headless, runs the trainer
and saves everything to Google Drive (a disconnect just means "run the cells again": `--resume`).

```bash
# needs the shared playground checked out next to this folder and its arena built
# (headless: maturin develop --release --no-default-features, run inside ../playground)
PYTHONPATH=../playground/python:. python3 train_ppo.py --save-dir runs/a  --steps 50e6
```

## What it trains
- **Policy**: `27 inputs -> tanh(Linear 256) -> GRU 256 -> 3 move means + shoot logit` (+ value head). ~0.4M parameters.
  The GRU is the agent's memory: with fog of war it can remember where the enemy was, where the walls are, how long ago it saw them.
- **Input**: `VectorObservationAdapter` (27 features). Inputs are fixed on purpose so every hall-of-fame snapshot can be replayed by the NumPy policy.
- **Exported agent is pure NumPy** (`mytrain/gru_policy.py`, verified against torch by `--self-check`): no PyTorch on the laptop,
  ~0.1 ms per move, so it never gets near the referee's 500 ms limit.

## Why it should learn to *win* (and not memorize one opponent)
Opponents come from a **hall of fame** of its own past versions (snapshot every 10 updates, newest 4 always kept, max 30):
35% of those games are against the 2 newest snapshots, the rest uniform over the whole hall; 20% of all games are against the
rule-based bot as a fixed anchor. Maps are random every episode, the learner plays both sides.

Reward (training only — arena rules and referee are untouched; see `mytrain/selfplay_env.py`):

| Term | Value | Why |
|---|---|---|
| hit dealt / hit taken | +15 / -10 | engine reward, attack is worth more than defence |
| win / loss | +100 / -100 | engine reward |
| time penalty | -0.02 per tick | stalling is never free (a 1500-tick stall costs -30) |
| draw / timeout | -10 engine, **-40 extra** | a draw is clearly worse than winning, closer to a loss than to a win |
| shaping `gamma*phi(s') - phi(s)` | `phi = -3*aim_error - 3*dist` | closing in and aiming is rewarded; potential-based, so it doesn't change which policy is optimal |

All knobs are constants at the top of `mytrain/selfplay_env.py` (and `--bot-prob`, `--latest-prob`, `--snapshot-every`, `--hof-size`, ... in `train_ppo.py`).

## Reading the log
```
upd  120/3815 | 1.47M steps | 25000 sps | win 0.41 loss 0.35 draw 0.24 (bot win 0.62) | ep_len 610 | pg .. v .. ent .. kl .. | roll 0.5s upd 0.4s | EVAL vs_bot 0.71 vs_old 0.58 *best
```
- `win/loss/draw` — the learner against its own hall of fame + bot over the last 400 episodes. In self-play this hovers near 50% *by design*;
  progress shows in `bot win`, `EVAL vs_old` (> 0.5 = beats an old snapshot) and falling `draw` / `ep_len`.
- `EVAL` — fixed maps (never used in training), both sides. `*best` = new best of `vs_bot + vs_old`.
- `ent` falling slowly is healthy; `kl` should stay around 0.001-0.02; if `draw` stays near 1.0 after a few million steps, raise `--ent-coef` or the shaping weights.

## Taking the result home
`runs/a/export_best/` and `export_latest/` hold `agent.py` + `policy.npz` (NumPy only).
```bash
PYTHONPATH=playground/python python3 playground/python/referee.py export_best/agent.py friend_agent.py --seeds 20
```
**Final choice between candidates is the referee's job**, not the training log: duel `export_best`, `export_latest` and a few `hof/snap_*.npz`
(wrap one in an `agent.py` like the exported one) against each other and against your friend's agent.

## Notes
- Throughput on an 8-core laptop, CPU only: arenas ~21k env-steps/s with 7 workers, PPO update on CPU ~2.4 s per 14k steps. On 12 vCPU + A100 expect roughly 25-30k steps/s overall (measure it: `sps` in the log).
- `--steps 50e6` is a first target, not a guarantee. Curves usually show whether more is worth it.
- Changing the rules (arena size, physics) invalidates old hall of fame/checkpoints — start a new `--save-dir`.

## Handing a run over for review (`bundle`)
The trainer writes `metrics.csv` (one row per update) and `config.json`, and refreshes `<save-dir>/bundle/` at every evaluation:
config, metrics, `export_best/`, `export_latest/` and up to 6 evenly spaced hall-of-fame snapshots (~15 MB in total).
Zip that folder (the Colab notebook has a cell that downloads it as `tank_bundle.zip`) and run

    PYTHONPATH=../playground/python:. python3 assess.py tank_bundle.zip [--maps 30]

`assess.py` prints the learning curve, trends of the last 20% vs the 20% before, fresh duels on unseen maps (latest/best vs the
rule-based bot, latest vs older snapshots) and a verdict: *not done yet* (still beats its own past -> `--resume` with more `--steps`),
*plateau* (strong vs the bot, flat curve -> go duel the real opponent), *stuck* (don't just run longer: entropy / draw rate / reward).
Needs the shared arena repo next to this one (same rules version; `assess.py` warns on a mismatch).

## Hardware adapts automatically
`--workers` defaults to (vCPUs - 1) and `--envs-per-worker` to `ceil(--target-envs / workers)` (176 arenas in total), so the PPO batch
(~22k samples per update) is the same on a 2-vCPU free runtime and on a 12-vCPU one: weaker hardware is just slower (`sps` in the log),
not a different training. The GPU is used if present, otherwise the update runs on the CPU.
