

import enum
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec


class StocksEnvV1(gym.Env):
    '''

    refer to https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter10/lib/environ.py
    '''
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    # spec = EnvSpec('drl_investment/StocksEnv-v1', entry_point="drl_investment.envs.stocks_v1:StocksEnvV1")

    def __init__(self, config: dict):
        '''
        actions: 
          0 - buy
          1 - skip
          2 - sell

        config:
          data: DataFrame
        '''
        data: pd.DataFrame = config['data']
        _data = data[['open', 'high', 'low', 'close']]
        _data['high'] = _data['high']/_data['open'] - 1.0
        _data['low'] = _data['low']/_data['open'] - 1.0
        _data['close'] = _data['close']/_data['open'] - 1.0
        _data['position'] = 0.0
        self._data = _data.to_numpy()

        # self._bars_count = config.get('bars_count', 10)
        # offset = bars_count - 1
        self._commission_perc = config.get('commission_perc', 0.01)
        self._reward_on_close = config.get('reward_on_close', True)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4, ), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def _get_obs(self):
        self._data[self._offset][4] = self._position
        return self._data[self._offset][1:]

    def _get_info(self):
        return {}
    
    def _cur_close(self):
        """
        Calculate real close price for the current bar
        """
        open = self._data[self._offset][0]
        rel_close = self._data[self._offset][3]
        return open * (1.0 + rel_close)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # self._offset = self.np_random.choice(self._data.shape[0])
        self._offset = 1
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
        done = False
        close = self._cur_close()
        if action == 0 and self._position == 0:
            self._position = 1
            self._open_price = close
            reward -= self._commission_perc
        if action == 2 and self._position == 1:
            reward -= self._commission_perc
            # done |= self._reset_on_close
            if self._reward_on_close:
                reward += 100.0 * (close / self._open_price - 1.0)
            self._position = 0.0
            self._open_price = 0.0

        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= self._data.shape[0]-1

        if self._position == 1 and not self._reward_on_close:
            reward += 100.0 * (close / prev_close - 1.0)

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, done, info


    def render(self):
        pass

    def close(self):
        pass