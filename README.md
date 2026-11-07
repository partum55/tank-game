# Private training code (PPO + GRU self-play)

Mine only - not part of the shared `playground` repo. It imports the shared SDK (`tank_env`, `referee`) from
`../playground/python` and ships exported agents that are **self-contained** (agent.py + policy.npz + tank_gru_policy.py,
NumPy only), so the shared referee can duel them without any of this code.

    cd private
    PYTHONPATH=../playground/python:. python3 train_ppo.py --self-check
    PYTHONPATH=../playground/python:. python3 train_ppo.py --save-dir runs/a --workers 11 --envs-per-worker 16
    pytest tests                       # torch tests skip without PyTorch

Colab: `colab/train_colab.ipynb` (shared repo from GitHub, this folder from Google Drive). Details: `TRAINING.md`.
