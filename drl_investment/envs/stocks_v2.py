

import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

LOG = logging.getLogger(__name__)

class StocksEnvV2(gym.Env):
    '''

    refer to https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter10/lib/environ.py
    '''
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, config: dict):
        '''
        actions:
          0 - buy
          1 - skip
          2 - sell

        config:
          data: DataFrame
        '''
        super().__init__()
        data: pd.DataFrame = config['data']
        _data = data[['open', 'high', 'low', 'close', 'amount', 'volume']]
        _data['position'] = np.float32(0.0)

        self._column = ['open', 'high', 'low', 'close', 'amount', 'volume', 'position']
        self._data = _data[self._column]

        self._commission_perc = config.get('commission_perc', 0.0)
        self._reward_on_close = config.get('reward_on_close', False)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self._column), ), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def _get_obs(self):
        self._data.iloc[self._offset, -1] = np.float32(self._position)
        return self._data.iloc[self._offset].to_numpy(dtype=np.float32)

    def _get_info(self):
        return {'offset': self._offset, 'observation': self._data.iloc[self._offset].to_numpy().tolist()}

    def _cur_close(self):
        return self._data.iloc[self._offset]['close']

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # self._offset = self.np_random.choice(self._data.shape[0])
        self._offset = 1
        self._data['position'] = np.float32(0.0)
        self._position = self.np_random.choice(2)
        if self._position == 0:
            self._open_price = 0.0
        if self._position == 1:
            self._open_price = self._cur_close()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        reward = 0.0
        terminated = False
        close = self._cur_close()
        if action == 0 and self._position == 0:
            self._position = 1
            self._open_price = close
            # LOG.error(f'buy: {close}')
            reward -= self._commission_perc
        if action == 2 and self._position == 1:
            # LOG.error(f'sell: {close}')
            reward -= self._commission_perc
            # done |= self._reset_on_close
            if self._reward_on_close:
                reward += 100.0 * (close / self._open_price - 1.0)
            self._position = 0
            self._open_price = 0.0

        self._offset += 1
        prev_close = close
        close = self._cur_close()
        terminated |= self._offset >= self._data.shape[0]-1
        truncated = terminated

        if self._position == 1 and not self._reward_on_close:
            reward += 100.0 * (close / prev_close - 1.0)

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info


    def render(self):
        pass

    def close(self):
        pass