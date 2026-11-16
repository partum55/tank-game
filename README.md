# tank-game: PPO + GRU self-play trainer

My trainer for the tank duel - separate from the shared `playground` repo (arena + referee). It imports the shared SDK (`tank_env`, `referee`) from
`../playground/python` and ships exported agents that are **self-contained** (agent.py + policy.npz + tank_gru_policy.py + tank_adapt.py,
NumPy only), so the shared referee can duel them without any of this code. Rules v1.2 (20 shots, rectangular tanks, kill-only reward, 36 inputs,
optional adaptation before each match): see `TRAINING.md`.

    cd tank-game        # next to ../playground (the shared arena repo)
    PYTHONPATH=../playground/python:. python3 train_ppo.py --self-check
    PYTHONPATH=../playground/python:. python3 train_ppo.py --save-dir runs/a --workers 11 --envs-per-worker 16
    pytest tests                       # torch tests skip without PyTorch

Review a run: the trainer keeps `<save-dir>/bundle/` fresh (~15 MB); zip it and run `python3 assess.py bundle.zip`.

`arena/` is a snapshot of the shared arena + referee (`../playground` HEAD, rules v1.2) so that Colab needs only this repo; the rules live in
`../playground`: change them there, commit, run `./sync_arena.sh`, commit the new `arena/`.

Colab: `colab/train_colab.ipynb` (clones the arena repo and this one from GitHub; Drive only holds checkpoints). Details: `TRAINING.md`.
