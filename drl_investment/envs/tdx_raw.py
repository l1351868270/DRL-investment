import gym
from gym import spaces
import logging
import pygame
import numpy as np
import pandas as pd

from drl_investment.data.tdx import DataRaw


LOG = logging.getLogger(__name__)


class TDXRawEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, render_mode=None, data: np.array = None, columns: list[str] = None, max_position=0, min_position=0):
        if data is None:
            raise Exception(f'data must not be None')
        if columns is None:
            raise Exception(f'index must not be None')
        
        self._min_len = 100
        self._len = data.shape[0]
        if self._len < self._min_len:
            raise Exception(f'data length must large than {self._min_len}')
        
        assert data.shape[1] == len(columns)

        self._index: int = 0
        self._data = data
        self._columns = columns
        self._max_position = max_position
        self._min_position = min_position

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1], ), dtype=np.float32)

        self.action_space = spaces.Discrete(3, start=-1)

    def _get_obs(self):
        return self._data[self._index]

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._index = self.np_random.integers(0, int(self._len*0.625))

        self._position = self.np_random.integers(self._min_position, self._max_position)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        terminated = self._index >= self._len
        if terminated:
            return None, None, True, False, None
        reward = 0
        if self.position == 0:
            reward = 0
        elif self.position > 0:
            reward = 0 if self._index==0 else self._data[self._index][0]/self._data[self._index-1][0]-1
            reward *= self._position
        elif self.position < 0:
            reward = 0 if self._index==0 else self._data[self._index][0]/self._data[self._index-1][0]-1
            reward *= self._position

        observation = self._get_obs()
        info = self._get_info()

        self._index += 1
        self._position += action
        return observation, reward, terminated, False, info

    def render(self):
        pass

    def close(self):
        pass