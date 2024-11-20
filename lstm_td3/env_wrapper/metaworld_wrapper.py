import gymnasium as gym
import numpy as np

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, ALL_V2_ENVIRONMENTS_GOAL_HIDDEN

from lstm_td3.env_wrapper.time_limit_wrapper import TimeLimit


class MetaWorldWrapper(gym.Wrapper):
    def __init__(self, env, action_repeat=2):
        super().__init__(env)
        self.env = env
        self.camera_name = "corner2"
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.env._freeze_rand_vec = False
        self.env.action_repeat = action_repeat
        self.env.render_mode = "rgb_array"


    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs = obs.astype(np.float32)
        self.env.step(np.zeros(self.env.action_space.shape))
        return obs, info

    def step(self, action):
        reward = 0
        for _ in range(self.env.action_repeat):
            obs, r, _, _, info = self.env.step(action.copy())
            reward += r
        obs = obs.astype(np.float32)
        return obs, reward, _, _, info


    @property
    def unwrapped(self):
        return self.env.unwrapped


    def render(self, *args, **kwargs):
        return self.env.render().copy()

def make_mw_env(task, seed=1):
    """
    Make Meta-World environment.
    """
    env_id = task.split("-", 1)[-1] + "-v2-goal-hidden"
    if not task.startswith('mw-') or not (env_id in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE or env_id in ALL_V2_ENVIRONMENTS_GOAL_HIDDEN):
        raise ValueError('Unknown task:', task)
    env = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_id](seed=seed)
    env = MetaWorldWrapper(env)
    env = TimeLimit(env, max_episode_steps=100)
    env.max_episode_steps = env._max_episode_steps
    return env
