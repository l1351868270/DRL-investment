import gymnasium as gym
from gymnasium import spaces
import logging
import numpy as np
import pandas as pd
# from gym.envs.registration import register

from drl_investment.data.tdx import DataRaw


LOG = logging.getLogger(__name__)


class TDXRawEnv(gym.Env):
    '''
    refer to: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
    '''
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, config: dict):
        self._data: np.array = config['data']
        self._columns: list[str] = config['columns']
        self._initial_position: int = config.get('initial_position', 1)
        self._max_position: int = config.get('max_position', 10)
        self._min_position: int = config.get('min_position', -10)
        
        self._min_len = 100
        self._len = self._data.shape[0]
        if self._len < self._min_len:
            raise Exception(f'data length must large than {self._min_len}')
        
        assert self._data.shape[1] == len(self._columns)

        self._index: int = 0

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._data.shape[1], ), dtype=np.float32)

        self.action_space = spaces.Discrete(3) # spaces.Discrete(3, start=-1) is not {-1, 0, 1}
        

    def _get_obs(self):
        return self._data[self._index]

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._index = self._begin_index = self.np_random.integers(0, 60)

        self._position = self._initial_position

        self._total_return = 0.0 # Total return until now

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        observation = self._get_obs()
        info = self._get_info()
    
        terminated = self._index >= self._len-1
        if terminated:
            return observation, 0.0, True, True, info
        
        if self._position > self._max_position:
            return observation, -2000.0, True, True, info
        if self._position < self._min_position:
            return observation, -2000.0, True, True, info
        
        reward = 0.0
        # ignore n+1 limit
        if self._position == 0:
            reward = 0
        elif self._position > 0:
            reward = 0 if self._index==0 else self._data[self._index][0]/self._data[self._index-1][0]-1
            reward *= self._position
        elif self._position < 0:
            reward = 0 if self._index==0 else self._data[self._index][0]/self._data[self._index-1][0]-1
            reward *= self._position
        
        self._index += 1
        self._position += action - 1

        # Total return less than -0.20, stop the game
        self._total_return += reward
        # if self._index > 1000:
        #     raise Exception(f'reward: {reward}, position: {self._position}, action: {action}, [self._index]: {self._index}')
        if self._total_return < -0.20:
            LOG.error((f'+++++++++++++++++++++reward: {reward}, total_return: {self._total_return}, position: {self._position}, action: {action}, [self._index]: {self._index}'))
            reward = -3000.0
            return observation, reward, True, True, info
        return observation, reward, terminated, terminated, info

    def render(self):
        pass

    def close(self):
        pass


# register(
#     id="drl_investment/TDXRaw-v0",
#     entry_point="drl_investment.envs.tdx_raw:TDXRawEnv",
# )