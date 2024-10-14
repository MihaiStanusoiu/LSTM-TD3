import gymnasium
from gymnasium import spaces
import numpy as np


class TimestepWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env: gymnasium.Env, use_relative_timestep=False):
        gymnasium.ObservationWrapper.__init__(self, env)
        self.use_relative_timestep = use_relative_timestep
        obs_space = self.env.observation_space
        assert isinstance(obs_space, spaces.Box), "Observation space must be continuous (spaces.Box)"

        # Add a dimension for the timestep
        low = np.append(obs_space.low, 0.0)
        high = np.append(obs_space.high, np.inf)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        if isinstance(self.env, DMCEnv):
            self.dt = self.env.control_timestep()

    def observation(self, observation):
        timestep = 0.0
        if isinstance(self.env, DMCEnv):
            if self.use_relative_timestep:
                timestep = self.env.control_timestep()
            else:
                timestep = self.env.physics.time()
        timestep_obs = np.append(observation, timestep)
        return timestep_obs