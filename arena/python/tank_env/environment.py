from typing import Optional, Tuple, Dict, Any, List

from .agent import sanitize_action

# Official rules: the arena is fixed. Trainers may shorten `max_steps`; the referee never does.
ARENA_WIDTH = 1200.0
ARENA_HEIGHT = 800.0
OFFICIAL_MAX_STEPS = 1500


class TankEnvironment:
    """
    Gym-like multi-agent 2D Tank Battle Environment.
    Wraps the Rust PyTankArena simulation engine.

    Every episode has a seeded, point-symmetric random map (see docs). `reset(seed)`
    reproduces the exact same map and spawns; `reset()` moves on to the next seed.
    Actions are sanitized here (NaN/inf/out-of-range can never reach the physics).
    """
    def __init__(self, max_steps: int = OFFICIAL_MAX_STEPS, seed: int = 0):
        import tank_arena  # lazy: agent processes only need the Python SDK, not the Rust extension

        self.max_steps = max_steps
        self.seed = seed
        self._arena = tank_arena.PyTankArena(
            width=ARENA_WIDTH,
            height=ARENA_HEIGHT,
            max_steps=max_steps,
            seed=seed,
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Starts a new episode and returns raw telemetry for both tanks."""
        return self._arena.reset(seed)

    def set_action(self, tank_id: int, action: List[float]):
        """Sets the action for a specific tank (1 or 2) without advancing the physical tick."""
        if not (1 <= tank_id <= 2):
            raise ValueError(f"tank_id must be 1 or 2, got {tank_id}")
        self._arena.set_action(tank_id, sanitize_action(action))

    def tick(self) -> Tuple[Dict[str, Any], Dict[str, Any], List[float], bool, Dict[str, Any]]:
        """Advances physical simulation by one tick using currently set actions."""
        return self._arena.tick()

    def step(
        self,
        action1: Optional[List[float]] = None,
        action2: Optional[List[float]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[float], bool, Dict[str, Any]]:
        """
        Advances the environment by one step.
        Supports both synchronous parallel stepping:
            step(action1, action2)
        and asynchronous stepping where one agent may still be thinking:
            step(action1=new_action, action2=None)  # tank 2 keeps its previous action
        """
        if action1 is not None:
            self._arena.set_action(1, sanitize_action(action1))
        if action2 is not None:
            self._arena.set_action(2, sanitize_action(action2))
        return self._arena.tick()

    def add_obstacle(self, x: float, y: float, width: float, height: float):
        """For experiments only: adds a wall on top of the generated map (referee never does)."""
        self._arena.add_obstacle(float(x), float(y), float(width), float(height))

    def clear_obstacles(self):
        """For experiments only: removes every obstacle of the current episode."""
        self._arena.clear_obstacles()

    def render(self) -> bool:
        """Renders current state in a native 60 FPS window. Returns False if user closed window."""
        return self._arena.render()

    def close(self):
        """Closes the rendering window and releases GUI resources."""
        self._arena.close()
