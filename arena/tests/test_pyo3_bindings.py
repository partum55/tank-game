import pytest
import tank_arena

def test_arena_lifecycle():
    arena = tank_arena.PyTankArena(width=800.0, height=600.0, max_steps=500, seed=123)
    obs1, obs2 = arena.reset()
    assert "x" in obs1 and "y" in obs1
    assert "hp" in obs1 and obs1["hp"] == 100.0
    assert "enemy_dist" in obs1
    assert "lidar_rays" in obs1
    assert len(obs1["lidar_rays"]) == 8

    act1 = [1.0, 0.0, 0.0, 0.0]
    act2 = [0.0, 1.0, 0.0, 1.0]
    next_obs1, next_obs2, rewards, done, info = arena.step(act1, act2)
    assert len(rewards) == 2
    assert not done
    assert info["step"] == 1

def test_parallel_action_setting():
    arena = tank_arena.PyTankArena(width=800.0, height=600.0, max_steps=500, seed=123)
    arena.reset()
    # Tank 1 acts asynchronously
    arena.set_action(1, [1.0, 0.0, 0.0, 0.0])
    # Tick simulation
    next_obs1, next_obs2, rewards, done, info = arena.tick()
    assert next_obs1["speed"] > 0.0
    assert next_obs2["speed"] == 0.0

def test_close_and_reopen():
    arena = tank_arena.PyTankArena(width=800.0, height=600.0, max_steps=100, seed=42)
    arena.reset()
    arena.close()

def test_obstacle_and_enemy_visibility():
    arena = tank_arena.PyTankArena(width=800.0, height=600.0, max_steps=100, seed=42)
    obs1, obs2 = arena.reset(seed=42)
    # Empty map: nothing between the tanks -> visible
    arena.clear_obstacles()
    obs1, obs2, _, _, _ = arena.step([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
    assert obs1["enemy_visible"] is True

    # Spawns are point-symmetric, so the line between tanks always crosses the centre,
    # which a central wall always blocks.
    arena.add_obstacle(380.0, 100.0, 40.0, 400.0)
    obs1, obs2, _, _, _ = arena.step([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
    # Now wall blocks line of sight
    assert obs1["enemy_visible"] is False
    assert obs2["enemy_visible"] is False

    # Clear obstacles
    arena.clear_obstacles()
    obs1, obs2, _, _, _ = arena.step([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
    assert obs1["enemy_visible"] is True


def test_seeded_reset_is_reproducible_and_exposes_map():
    arena = tank_arena.PyTankArena(width=800.0, height=600.0, max_steps=100, seed=1)
    a1, a2 = arena.reset(seed=7)
    b1, b2 = arena.reset(seed=7)
    assert a1 == b1 and a2 == b2
    assert all(len(o) == 4 for o in a1["obstacles"])
    assert a1["enemy_visible"] in (True, False) and a1["enemy_last_seen"] == 0.0
    # reset() without a seed moves on to a different episode
    c1, _ = arena.reset()
    d1, _ = arena.reset()
    assert (c1["x"], c1["y"], c1["obstacles"]) != (d1["x"], d1["y"], d1["obstacles"])
