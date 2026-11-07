# tank-game: PPO + GRU self-play trainer

My trainer for the tank duel - separate from the shared `playground` repo (arena + referee). It imports the shared SDK (`tank_env`, `referee`) from
`../playground/python` and ships exported agents that are **self-contained** (agent.py + policy.npz + tank_gru_policy.py,
NumPy only), so the shared referee can duel them without any of this code.

    cd tank-game        # next to ../playground (the shared arena repo)
    PYTHONPATH=../playground/python:. python3 train_ppo.py --self-check
    PYTHONPATH=../playground/python:. python3 train_ppo.py --save-dir runs/a --workers 11 --envs-per-worker 16
    pytest tests                       # torch tests skip without PyTorch

Colab: `colab/train_colab.ipynb` (clones the arena repo and this one from GitHub; Drive only holds checkpoints). Details: `TRAINING.md`.
