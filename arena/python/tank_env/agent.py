"""
The one contract between a player and the playground.

An AGENT is any object with:
    act(obs: dict) -> sequence of 4 floats   [thrust, turn, turret_turn, shoot]
    reset()                                   (optional, called before every match)

An agent FILE is a `.py` file that defines `make_agent()` returning such an object.
Model, weights and input preprocessing all live in your file (or next to it) -
the playground knows nothing about NumPy, PyTorch, adapters or checkpoint formats.
Load files relative to `Path(__file__).parent`.
"""
import importlib.util
import math
import os
import sys
from typing import Any, List

from .bots import RandomBot, RuleBasedBot


def sanitize_action(action: Any) -> List[float]:
    """
    Trust boundary: whatever an agent returns becomes exactly 4 finite floats in [-1, 1].
    NaN/inf -> 0, extra values dropped, missing values -> 0 (no shoot).
    """
    try:
        values = list(action)[:4]
    except TypeError:
        values = []
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = 0.0
        out.append(max(-1.0, min(1.0, f)) if math.isfinite(f) else 0.0)
    return out + [0.0] * (4 - len(out))


class MLPAgent:
    """A SimpleMLP + an ObservationAdapter packaged as an agent."""

    def __init__(self, model, adapter):
        if model.input_dim != adapter.output_dim:
            raise ValueError(
                f"model expects {model.input_dim} inputs but "
                f"{type(adapter).__name__} produces {adapter.output_dim}"
            )
        self.model = model
        self.adapter = adapter

    def act(self, obs: dict) -> List[float]:
        return self.model.forward(self.adapter.extract(obs))

    @classmethod
    def load(cls, npz_path, adapter) -> "MLPAgent":
        from .neural_net import SimpleMLP

        return cls(SimpleMLP.load(str(npz_path)), adapter)


def reset_agent(agent) -> None:
    fn = getattr(agent, "reset", None)
    if callable(fn):
        fn()


def load_agent(spec: str):
    """`rule_based` | `random` | path to an agent `.py` file."""
    if spec == "rule_based":
        return RuleBasedBot()
    if spec == "random":
        return RandomBot()
    if not spec.endswith(".py"):
        raise ValueError(
            f"Unknown agent '{spec}'. Use 'rule_based', 'random' or a .py file "
            "that defines make_agent() (see examples/template_agent.py)."
        )
    path = os.path.abspath(spec)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Agent file not found: {spec}")

    module_name = f"_tank_agent_{abs(hash(path))}"
    mod_spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(mod_spec)
    sys.modules[module_name] = module
    folder = os.path.dirname(path)
    sys.path.insert(0, folder)  # agent may import helper modules that sit next to it
    try:
        mod_spec.loader.exec_module(module)
        if not hasattr(module, "make_agent"):
            raise ValueError(f"{spec} must define make_agent()")
        agent = module.make_agent()
    finally:
        sys.path.remove(folder)
    if not callable(getattr(agent, "act", None)):
        raise ValueError(f"make_agent() in {spec} must return an object with act(obs)")
    return agent
