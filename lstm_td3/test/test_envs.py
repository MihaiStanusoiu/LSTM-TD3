import gymnasium
import numpy as np
from gymnasium.wrappers import TimeLimit


class TestEnv1(gymnasium.Env):
    dt = 1.
    def __init__(self):
        super().__init__()
        self.dt = 1.
        self.action_space = gymnasium.spaces.Box(low=0, high=0, shape=(1,), dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(low=0, high=0, shape=(1,), dtype=np.float32)

    def reset(self):
        return np.zeros(shape=(1,)), {}

    def step(self, action):
        return np.array([0], dtype=np.float32), 1, True, False, {}

class TestEnv2:
    dt = 1.

    def __init__(self):
        self.dt = 1.
        self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(low=0, high=0, shape=(1,), dtype=np.float32)

    def reset(self):
        return np.zeros(shape=(1,))

    def step(self, action):
        return np.array([0], dtype=np.float32), np.clip(action, -1, 1), True, {}

def make_test_env_vf():
    return TimeLimit(TestEnv1(), max_episode_steps=1)

def make_test_env_pi():
    return TestEnv2()