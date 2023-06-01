

import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec

LOG = logging.getLogger(__name__)

class StocksEnvV4(gym.Env):
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
        _data = data[['open', 'high', 'low', 'close', 'amount', 'volume']]
        _data['position'] = np.float32(0.0)
        open = data.open
        high = _data.high
        low = _data.low
        close = _data.close
        amount = _data.amount
        volume = _data.volume
        returns = _data.close.pct_change()
        vwap = (_data.volume*_data.close)/_data.volume
        
        # Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
        alpha_001_power = _data.close
        alpha_001_power[returns < 0] = returns.rolling(20).std()
        alpha_001_signed_power = np.sign(alpha_001_power) * np.power(alpha_001_power, 2.0)
        alpha_001_Ts_ArgMax = alpha_001_signed_power.rolling(5).apply(np.argmax)
        _data['alpha_001'] = alpha_001_Ts_ArgMax.rank(pct=True) - 0.5
        LOG.debug(f'alpha_001: {_data.alpha_001}')
        # Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
        _data['alpha_002'] = -1 * (_data['volume'].apply(np.log).diff(periods=2).rank(pct=True).rolling(6).corr(((_data.close - _data.open)/_data.open).rank(pct=True)))
        LOG.debug(f'alpha_002: {_data.alpha_002}')
        # Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))
        _data['alpha_003'] = -1 * (_data.open.rank(pct=True).rolling(10).corr(_data.volume.rank(pct=True)))
        LOG.debug(f'alpha_003: {_data.alpha_003}')
        # Alpha#4: (-1 * Ts_Rank(rank(low), 9))
        _data['alpha_004'] = -1 *  _data['low'].rank(pct=True).rolling(9).rank(pct=True)
        LOG.debug(f'alpha_004: {_data.alpha_004}')
        # Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
        _data['alpha_005'] = ((_data.open - vwap.rolling(10).sum()/10).rank(pct=True)) * (-1 * ((_data.close - vwap).rank(pct=True).abs())) 
        LOG.debug(f'alpha_005: {_data.alpha_005}')
        # Alpha#6: (-1 * correlation(open, volume, 10))
        _data['alpha_006'] = -1 * (_data.open.rolling(10).corr(_data.volume))
        LOG.debug(f'alpha_006: {_data.alpha_006}')
        # Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
        alpha_007_inner = -1 * (_data.close.diff(periods=7).abs().rolling(60).rank(pct=True)) * (_data.close.diff(7).apply(np.sign))
        alpha_007_inner[_data.volume.rolling(20).mean() >= _data.volume] = -1 * 1
        _data['alpha_007'] = alpha_007_inner
        LOG.debug(f'alpha_007: {_data.alpha_007}')
        # Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))
        _data['alpha_008'] = -1 * ((_data.open.rolling(5).sum() * returns.rolling(5).sum() - _data.open.rolling(5).sum() * returns.rolling(5).sum().shift(periods=10)).rank(pct=True))
        LOG.debug(f'alpha_008: {_data.alpha_008}')
        # Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
        alpha_009_inner = -1 * (_data.close.diff(periods=1))
        alpha_009_inner[_data.close.diff(periods=1).rolling(5).max() < 0] = _data.close.diff(periods=1)
        alpha_009_inner1 = alpha_009_inner
        alpha_009_inner1[_data.close.diff(periods=1).rolling(5).min() > 0] = _data.close.diff(periods=1)
        _data['alpha_009'] = alpha_009_inner1
        LOG.debug(f'alpha_009: {_data.alpha_009}')
        # Alpha#10: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
        alpha_010_inner = -1 * (_data.close.diff(periods=1))
        alpha_010_inner[_data.close.diff(periods=1).rolling(4).max() < 0] = _data.close.diff(periods=1)
        alpha_010_inner1 = alpha_010_inner
        alpha_010_inner1[_data.close.diff(periods=1).rolling(4).min() > 0] = _data.close.diff(periods=1)
        _data['alpha_010'] = alpha_010_inner1.rank(pct=True)
        LOG.debug(f'alpha_010: {_data.alpha_010}')
        # Alpha#11: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
        _data['alpha_011'] = ((vwap - close).rolling(3).max().rank(pct=True) + ((vwap - close).rolling(3).min().rank(pct=True))) * (volume.diff(3).rank())
        LOG.debug(f'alpha_011: {_data.alpha_011}')
        # Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1))) 
        _data['alpha_012'] = (volume.diff(periods=1).apply(np.sign)) * (-1 * (close.diff(periods=1)))
        LOG.debug(f'alpha_012: {_data.alpha_012}')
        # Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5))) 
        _data['alpha_013'] = -1 * ((close.rank(pct=True)).rolling(5).cov(volume.rank(pct=True)))
        LOG.debug(f'alpha_013: {_data.alpha_013}')
        # Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)) 
        _data['alpha_014'] = (-1 * (returns.diff(3).rank(pct=True))) * (open.rolling(10).corr(volume))
        LOG.debug(f'alpha_014: {_data.alpha_014}')
        # Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)) 
        _data['alpha_015'] = -1 * (((high.rank(pct=True).rolling(3).corr(volume.rank(pct=True))).rank(pct=True)).rolling(3).sum())
        LOG.debug(f'alpha_015: {_data.alpha_015}')
        # Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), 5))) 
        _data['alpha_016'] = -1 * ((high.rank(pct=True).rolling(5).cov(volume.rank(pct=True))).rank(pct=True))
        LOG.debug(f'alpha_016: {_data.alpha_016}')
        # Alpha#17: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5))) 
        _data['alpha_017'] = ((-1 * ((close.rolling(10).rank(pct=True)).rank(pct=True))) * (close.diff(periods=1).diff(periods=1).rank(pct=True))) * \
                             ((volume / volume.rolling(20).mean()).rolling(5).rank(pct=True))
        LOG.debug(f'alpha_017: {_data.alpha_017}')
        # Alpha#18: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10)))) 
        _data['alpha_018'] = -1 * ((((close - open).abs()).rolling(5).std() + (close - open) + close.rolling(10).corr(open)).rank(pct=True))
        LOG.debug(f'alpha_018: {_data.alpha_018}')
        # Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250))))) 
        _data['alpha_019'] = (-1 * ((close - close.shift(periods=7) + close.diff(periods=7)).apply(np.sign))) * (1 + (1 + returns.rolling(250).sum()).rank(pct=True))
        LOG.debug(f'alpha_019: {_data.alpha_019}')
        # Alpha#20: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
        _data['alpha_020'] = (-1 * ((open - high.shift(periods=1)).rank(pct=1))) * ((open - close.shift(1)).rank(pct=True)) * ((open - low.shift(1)).rank(pct=True))
        LOG.debug(f'alpha_020: {_data.alpha_020}')
        
        # self._column = ['open', 'high', 'low', 'close', 'amount', 'volume', 'position']
        self._column = [             'alpha_001', 'alpha_002', 'alpha_003', 'alpha_004', 'alpha_005', 'alpha_006', 'alpha_007', 'alpha_008', 'alpha_009',  
                        'alpha_010', 'alpha_011', 'alpha_012', 'alpha_013', 'alpha_014', 'alpha_015', 'alpha_016', 'alpha_017', 'alpha_018', 'alpha_019', 
                        'alpha_020', ]
        self._data = _data[self._column]

        self._bars_count = config.get('bars_count', 10)
        self._offset = self._bars_count - 1
        self._commission_perc = config.get('commission_perc', 0.01)
        self._reward_on_close = config.get('reward_on_close', False)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._bars_count, len(self._column), ), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def _get_obs(self):
        self._data.iloc[self._offset, -1] = self._position
        return self._data.iloc[self._offset-self._bars_count+1:self._offset+1].to_numpy()

    def _get_info(self):
        return {'offset': self._offset, 'bars_count': self._bars_count, 'observation': self._get_obs().tolist()}
    
    def _cur_close(self):
        """
        Calculate real close price for the current bar
        """
        return self._data.iloc[self._offset]['close']
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # self._offset = self.np_random.choice(self._data.shape[0])
        self._offset = self._bars_count - 1
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
        done = False
        close = self._cur_close()
        if action == 0 and self._position == 0:
            self._position = 1
            self._open_price = close
            LOG.error(f'buy: {close}')
            reward -= self._commission_perc
        if action == 2 and self._position == 1:
            LOG.error(f'sell: {close}')
            reward -= self._commission_perc
            # done |= self._reset_on_close
            if self._reward_on_close:
                reward += 100.0 * (close / self._open_price - 1.0)
            self._position = 0
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