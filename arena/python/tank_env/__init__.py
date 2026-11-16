from .environment import TankEnvironment, OFFICIAL_MAX_STEPS
from .adapters import ObservationAdapter, VectorObservationAdapter, LidarObservationAdapter
from .agent import MLPAgent, load_agent, reset_agent, sanitize_action
from .bots import RuleBasedBot, RandomBot
from .neural_net import SimpleMLP

# Bump when physics / map generation / contract change. Both players must use the same one.
RULES_VERSION = "1.2"

__all__ = [
    "RULES_VERSION",
    "OFFICIAL_MAX_STEPS",
    "TankEnvironment",
    "ObservationAdapter",
    "VectorObservationAdapter",
    "LidarObservationAdapter",
    "MLPAgent",
    "load_agent",
    "reset_agent",
    "sanitize_action",
    "RuleBasedBot",
    "RandomBot",
    "SimpleMLP",
]
