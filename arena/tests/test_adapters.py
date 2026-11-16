import numpy as np
from tank_env.environment import TankEnvironment
from tank_env.adapters import VectorObservationAdapter, LidarObservationAdapter
from tank_env.bots import RuleBasedBot, RandomBot

def test_observation_adapters():
    env = TankEnvironment(seed=99)
    obs1_raw, obs2_raw = env.reset()

    vec_adapter = VectorObservationAdapter()
    v1 = vec_adapter.extract(obs1_raw)
    assert isinstance(v1, np.ndarray)
    assert v1.shape == (27,)
    assert not np.isnan(v1).any()

    lidar_adapter = LidarObservationAdapter(num_rays=8)
    l1 = lidar_adapter.extract(obs1_raw)
    assert isinstance(l1, np.ndarray)
    assert l1.shape == (15,)  # 8 rays + 7 state floats
    assert not np.isnan(l1).any()
    assert v1.shape == (vec_adapter.output_dim,) and l1.shape == (lidar_adapter.output_dim,)

def test_rule_based_bot():
    env = TankEnvironment(seed=99)
    bot = RuleBasedBot()
    obs1, obs2 = env.reset()
    action = bot.act(obs1)
    assert len(action) == 4
    assert all(-1.0 <= a <= 1.0 for a in action)

def test_random_bot():
    bot = RandomBot()
    action = bot.act({})
    assert len(action) == 4
    assert all(-1.0 <= a <= 1.0 for a in action)

def test_asynchronous_step_interface():
    env = TankEnvironment(seed=99)
    env.reset()
    # Tank 1 acts, Tank 2 is None (stays as previous action or noop)
    obs1, obs2, rewards, done, info = env.step(action1=[0.5, 0.0, 0.0, 0.0], action2=None)
    assert obs1["speed"] > 0.0
