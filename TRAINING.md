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

## Where the arenas run: `--env`
- `--env gpu` (default when CUDA is present): a tensor re-implementation of the arena (`mytrain/gpu_arena.py`) steps `--gpu-envs`
  (2048) games at once, the opponents (hall-of-fame snapshots + rule-based bot) run in tensors too, nothing crosses to the CPU per step.
  It copies the Rust tick order and formulas; `tests/test_gpu_parity.py` replays random games on both engines and compares every tick
  (features, rewards, game end). Maps and spawns come from a bank generated by the Rust engine, so the distribution is the official one.
  Evaluation and every duel still run on the Rust engine, so a simulator mismatch would show up as a gap between training and `EVAL`.
- `--env cpu`: the Rust engine in worker processes (CPU-bound, ~5k steps/s on 12 vCPU).

GPU-env specifics: only `--slots` (6) hall-of-fame snapshots are "on stage" at once (a new snapshot replaces the oldest slot, otherwise one
slot is re-drawn at random per refresh), so an env can meet a changed opponent mid-episode. Known deliberate difference from Rust: several
bullets hitting one tank in the same tick all count (Rust lets a bullet after the killing one pass through) - last tick of an episode only.

## GPU sizing (`--gpu-envs 0`, the default)
The simulator step costs ~1.7k tiny kernels, so its time per step is almost independent of the number of arenas until the GPU
saturates: throughput grows ~linearly with arenas. `autotune_envs` doubles the number of arenas (1024, 2048, ...) until a size is
less than 15% faster than the previous one (or out of memory, or `--max-envs`). Then the number of arenas is capped by learning:
`steps / (rollout * min_updates)` (default `--min-updates 400`), because PPO learns per update and a 100k-arena batch gives only a handful
of updates for a 200M budget. Rule of thumb: arenas ~ budget / 50k. To use the GPU more, raise `--steps` (e.g. 1000e6 -> ~16k arenas);
`--gpu-envs N` overrides everything. TF32 matmuls are enabled on CUDA (the opponents' GRUs run ~several x faster on an A100).

## Map variety (`--map-mix`, default 0.3)
Next to the official maps the GPU arena trains on `mytrain/mapmix.py` layouts: open, sparse, dense, close-start, free spawns/headings
(same guarantees: point-symmetric, gaps >= 3 radii, spawns clear of walls). Goal: movement and aiming that do not depend on one layout.
Training-only: EVAL and the referee use the official Rust maps, so a harmful mix shows up as a gap between training and `EVAL`
(`--map-mix 0` switches it off).

## Measured on an A100
~52k env-steps/s at 2048 arenas x 128 steps (5 s per update). A run with 190 updates (50M steps) is far too short: judge a run only after
>= 250 updates (`assess.py` says "too early" otherwise). Recommended: `--rollout 64 --steps 150e6` (~570 updates).

## The agent samples, like the policy that was trained (important)
PPO trains a stochastic policy: Gaussian noise on the three moves (`log_std`), Bernoulli shot. Such a policy may *depend on its own noise*.
Measured on a real run: the same weights scored **0.89 vs the bot when sampling and 0.04 when playing mean/threshold** (Rust engine; the
GPU arena agreed: 0.86 vs 0.06 - which also confirms the simulator end to end). So `log_std` is exported with the weights, and
`NumpyGRUPolicy` (exports, evaluation, hall-of-fame opponents on CPU and GPU) samples with it, with a fixed seed that `reset()` re-applies
(a match is reproducible). Exports without `log_std` (older bundles) fall back to the deterministic mean. `NumpyGRUPolicy(params, deterministic=True)` forces it.

## Throughput facts (A100, 12 vCPU)
Simulator only: 28k sps at 1k arenas ... 1.67M sps at 65k (time per step ~37 ms regardless: launch-bound). A full update (2048-4096 arenas):
rollout 2.4 s + PPO 0.6 s. Evaluation (Rust, 48 games) took 16 s in one process; `--eval-procs` (default vCPUs-1) now spreads it.

## CUDA graph rollout (`--cuda-graph auto|on|off`)
`mytrain/rollout.py`: one rollout step (policy -> arena -> buffers) runs over static tensors; `static_env_step` copies fresh state back
into the original tensors so a captured graph can be replayed (test: trajectories are bit-identical to the plain step). The step index is a
device tensor, randomness uses the CUDA generator (graph-safe), and the simulator contains no host syncs. At start the trainer times
the eager loop and the replay, prints both, and keeps the graph only if it works and is faster; any error falls back to eager.
Expected: the ~37 ms/step launch overhead collapses, so rollouts get several times faster (not measured yet - no GPU where this was written).

## Choosing the final agent: `pick_best.py`
`python3 pick_best.py bundle.zip [--maps 8] [--extra other/policy.npz] [--out runs/final]` plays a round-robin of every export and
hall-of-fame snapshot in the bundle (plus the rule-based bot and any `--extra` policies) on the Rust engine, in parallel, and writes the
highest-ranked one as a ready-to-duel agent folder. Training's own `export_best` is picked from 12-24 games and can be a lucky one; this
tournament is the better selector. Exports without `log_std` (older bundles) are flagged because they play deterministically.

## v2 inputs and reward (31 features)
Observed on the first strong agent: it hugged walls, shot at where the enemy *was*, and moved jerkily. Causes and fixes:
- **No lead**: the model had no enemy velocity (and the GRU never inferred it). Now `FeatureExtractor` (stateful) adds the enemy velocity
  (change of its position between two consecutive visible ticks, in the own-heading frame) and sin/cos of the aim error to the lead point
  (`lead_angle_error`: bullet intercept, law of sines). The GPU arena computes the same (`tests/test_gpu_parity.py` replays Rust games and
  compares all 31 inputs). Exports with 27 inputs still load (`FeatureExtractor(27)`).
- **Wall hugging**: the distance shaping pulls towards the (last known) enemy through walls. Added `STUCK_PENALTY` (0.05 per tick of driving
  into something: commanded speed > 1 px/tick but moved < 30% of it; GPU arena only) and the aim shaping now uses the lead error.
- **Jitter**: ablation on the first agent: moves need no exploration noise (0.86 exact, 0.91 with 0.25, 0.83 with the full 0.5); only the
  *shot* must stay sampled (its probability sat at ~0.45 on every tick). So moves are played with `MOVE_NOISE_SCALE` (0.5) x exp(log_std).
