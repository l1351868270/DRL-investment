import abc
import pandas as pd
import numpy as np
from typing import Union, Any


class DataBase(abc.ABC):
    def __init__(self):
        return

    @property
    def raw(self):
        return

    @property
    @abc.abstractmethod
    def alpha101(self):
        return

    @property
    @abc.abstractmethod
    def alpha158(self):
        return

    @property
    @abc.abstractmethod
    def alpha360(self):
        return


class Alpha101(object):
    def __init__(self, data: pd.DataFrame=None):
        self.open: pd.Series = data.open 
        self.high: pd.Series = data.high
        self.low: pd.Series = data.low
        self.close: pd.Series = data.close
        self.amount: pd.Series = data.amount
        self.volume: pd.Series = data.volume
        self.returns: pd.Series = data.close.pct_change() # daily close-to-close returns
        self.vwap: pd.Series = (data.volume*data.close)/data.volume

    def abs(self, x: pd.Series) -> pd.Series:
        return x.abs()
    
    def log(self, x: pd.Series) -> pd.Series:
        return x.apply(np.log)
    
    def sign(self, x: pd.Series) -> pd.Series:
        return x.apply(np.sign)
    
    def rank(self, x: pd.Series) -> pd.Series:
        # rank(x) = cross-sectional rank
        return x.rank(pct=True)
    
    def delay(self, x: pd.Series, d: int) -> pd.Series:
        # delay(x, d) = value of x d days ago
        return x.shift(periods=d)
    
    def correlation(self, x: pd.Series, y: pd.Series, d: int) -> pd.Series:
        # correlation(x, y, d) = time-serial correlation of x and y for the past d days
        return x.rolling(d).corr(y)
    
    def covariance(self, x: pd.Series, y: pd.Series, d: int) -> pd.Series:
        # covariance(x, y, d) = time-serial covariance of x and y for the past d days
        return x.rolling(d).cov(y)
    
    def scale(self, x: pd.Series, a = 1.0) -> pd.Series:
        # scale(x, a) = rescaled x such that sum(abs(x)) = a (the default is a = 1)
        return x * (a / (x.abs().sum()))
        
    def delta(self, x: pd.Series, d: int) -> pd.Series:
        # delta(x, d) = today’s value of x minus the value of x d days ago
        return x.diff(d)
    
    def signedpower(self, x: pd.Series, a) -> pd.Series:
        # signedpower(x, a) = x^a
        # sign(x) * (abs(x)**a)
        return (x.apply(np.sign)) * (x.abs().pow(a))
        
    def decay_linear(self, x: pd.Series, d: int) -> pd.Series:
        # decay_linear(x, d) = weighted moving average over the past d days with linearly decaying weights d, d – 1, …, 1 (rescaled to sum up to 1)
        return x.rolling(d).apply(lambda x: ((np.arange(d, 0, -1)*x)/(np.arange(d, 0, -1).sum())).sum())
    
    def indneutralize(self, x: pd.Series, g) -> pd.Series:
    # indneutralize(x, g) = x cross-sectionally neutralized against groups g (subindustries, industries, sectors, etc.), i.e., x is cross-sectionally demeaned within each group g
        raise Exception('Have not complete')
    
    def ts_O(self, x, d) -> pd.Series:
        # ts_{O}(x, d) = operator O applied across the time-series for the past d days; non-integer number of days d is converted to floor(d)
        raise Exception('Have not complete')
    
    def ts_min(self, x: pd.Series, d: int) -> pd.Series:
        # ts_min(x, d) = time-series min over the past d days
        return x.rolling(d).min()
    
    def ts_max(self, x: pd.Series, d: int) -> pd.Series:
        # ts_max(x, d) = time-series max over the past d days
        return x.rolling(d).max()

    def ts_argmax(self, x: pd.Series, d: int) -> pd.Series:
        # ts_argmax(x, d) = which day ts_max(x, d) occurred on
        return x.rolling(d).apply(np.argmax)
    
    def ts_argmin(self, x: pd.Series, d: int) -> pd.Series:
        # ts_argmin(x, d) = which day ts_min(x, d) occurred on
        return x.rolling(d).apply(np.argmin)
    
    def ts_rank(self, x: pd.Series, d: int) -> pd.Series:
        # ts_rank(x, d) = time-series rank in the past d days
        return x.rolling(d).rank(pct=True)
    
    def min(self, x: pd.Series, d: int) -> pd.Series:
        # min(x, d) = ts_min(x, d) 
        return self.ts_min(x, d)
    
    def max(self, x: pd.Series, d: int) -> pd.Series:
        # max(x, d) = ts_max(x, d) 
        return self.ts_max(x, d)
    
    def sum(self, x: pd.Series, d: int) -> pd.Series:
        # sum(x, d) = time-series sum over the past d days
        return x.rolling(d).sum()

    def product(self, x: pd.Series, d: int) -> pd.Series:
        # product(x, d) = time-series product over the past d days
        return x.rolling(d).apply(np.product)
    
    def stddev(self, x: pd.Series, d: int) -> pd.Series:
        # stddev(x, d) = moving time-series standard deviation over the past d days
        return x.rolling(d).std()
    
    def adv(self, x: pd.Series, d: int) -> pd.Series:
        # adv{d} = average daily dollar volume for the past d days
        return x.rolling(d).mean()
    
    def condition(self, con: pd.Series, x: Union[pd.Series, int, float], y: Union[pd.Series, int, float]) -> pd.Series:
        assert isinstance(x, pd.Series) or isinstance(y, pd.Series), 'x or y must has one pd.Series'
        if (not isinstance(x, pd.Series)) and (not isinstance(y, pd.Series)):
            inner = pd.Series(x, np.float32(y))
            inner[con] = np.float32(x)
        elif not isinstance(y, pd.Series):
            inner: pd.Series = x
            inner[~con] = y
        else:
            inner: pd.Series = y
            inner[con] = x

        return inner

    def alpha_001(self) -> pd.Series:
        # Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
        return self.rank(self.ts_argmax(self.signedpower(self.condition(self.returns, self.stddev(self.returns, 20), self.close), 2), 5)) - 0.5
    
    def alpha_002(self) -> pd.Series:
        # Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6)) 
        return -1 * self.correlation(self.rank(self.delta(self.log(self.volume), 2)), self.rank((self.close - self.open) / self.open) , 6)
    
    def alpha_003(self) -> pd.Series:
       # Alpha#3: (-1 * correlation(rank(open), rank(volume), 10)) 
       return -1 * self.correlation(self.rank(self.open), self.rank(self.volume), 10)
    
    def alpha_004(self) -> pd.Series:
        # Alpha#4: (-1 * Ts_Rank(rank(low), 9)) 
        return -1 * self.ts_rank(self.rank(self.low), 9)
    
    def alpha_005(self) -> pd.Series:
        # Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap))))) 
        return self.rank(self.open - self.sum(self.vwap, 10) / 10) * (-1 * self.abs(self.rank(self.close - self.vwap)))
    
    def alpha_006(self) -> pd.Series:
        # Alpha#6: (-1 * correlation(open, volume, 10)) 
        return -1 * self.correlation(self.open, self.volume, 10)
    
    def alpha_007(self) -> pd.Series:
        # Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
        return self.condition(self.adv(20) < self.volume, -1 * self.ts_rank(self.abs(self.delta(self.close, 7)), 60), -1 * 1)
    
    def alpha_008(self) -> pd.Series:
        # Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))) 
        inner = self.sum(self.open, 5) * self.sum(self.returns, 5)
        return -1 * self.rank(inner - self.delay(inner, 10))
    
    def alpha_009(self):
        # Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))) 
        inner = self.delta(self.close, 1)
        return self.condition(0 < self.ts_min(inner, 5), inner, self.condition(self.ts_max(inner, 5) < 0, inner, -1 * inner))
        
    def alpha_010(self) -> pd.Series:
        # Alpha#10: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
        inner = self.delta(self.close, 1)
        return self.condition(0 < self.ts_min(inner, 4), inner, self.condition(self.ts_max(inner, 4) < 0, inner, -1 * inner))
    
    def alpha_011(self) -> pd.Series:
        # Alpha#11: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3))) 
        return (self.rank(self.ts_max(self.vwap - self.close, 3)) + self.rank(self.ts_min(self.vwap - self.close, 3))) * self.rank(self.delta(self.volume, 3))
    
    def alpha_012(self) -> pd.Series:
        # Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1))) 
        return self.sign(self.delta(self.volume, 1)) * -1 * self.delta(self.close, 1)
    
    def alpha_013(self) -> pd.Series:
        # Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5))) 
        return -1 * self.rank(self.covariance(self.rank(self.close), self.rank(self.volume), 5))
    
    def alpha_014(self) -> pd.Series:
        # Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)) 
        return -1 * self.rank(self.delta(self.returns, 3)) * self.correlation(self.open, self.volume, 10)

    def alpha_015(self) -> pd.Series:
        # Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)) 
        return -1 * self.sum(self.rank(self.correlation(self.rank(self.high), self.rank(self.volume), 3)), 3)
    
    def alpha_016(self) -> pd.Series:
        # Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), 5))) 
        return -1 * self.rank(self.covariance(self.rank(self.high), self.rank(self.volume), 5))
    
    def alpha_017(self) -> pd.Series:
        # Alpha#17: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5))) 
        return -1 * self.rank(self.ts_rank(self.close, 10)) * self.rank(self.delta(self.delta(self.close, 1), 1)) * self.rank(self.ts_rank(self.volume / self.adv(20), 5))
    
    def alpha_018(self) -> pd.Series:
        # Alpha#18: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10)))) 
        return -1 * self.rank(self.stddev(self.abs(self.close - self.open), 5) + (self.close - self.open) + self.correlation(self.close, self.open, 10))

    def alpha_019(self) -> pd.Series:
        # Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250))))) 
        return -1 * self.sign((self.close - self.delay(self.close, 7) + self.delta(self.close, 7))) * (1 + self.rank(1 + self.sum(self.returns, 250)))
    
    def alpha_020(self) -> pd.Series:
        # Alpha#20: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
        return -1 * self.rank(self.open - self.delay(self.high, 1)) * self.rank(self.open - self.delay(self.close, 1)) *self.rank(self.open - self.delay(self.low, 1))

    def alpha_021(self) -> pd.Series:
        # Alpha#21: ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1)))) 
        return self.condition(self.sum(self.close, 8) / 8 + self.stddev(self.close, 8) < self.sum(self.close, 2) / 2, -1, self.condition(self.sum(self.close, 2) / 2 < self.sum(self.close, 8) - self.stddev(self.close, 8), 1, self.condition(1 <= self.volume / self.adv(20), 1, -1)))

    def alpha_022(self) -> pd.Series:
        # Alpha#22: (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20)))) 
        return -1 * self.delta(self.correlation(self.high, self.volume, 5), 5) * self.rank(self.stddev(self.close, 20))
        
    def alpha_023(self) -> pd.Series:
        # Alpha#23: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0) 
        return self.condition(self.sum(self.high, 20) / 20 < self.high, -1 * self.delta(self.high, 2), 0)
    
    def alpha_024(self) -> pd.Series:
        # Alpha#24: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))
        return self.condition(self.delta(self.sum(self.close, 100) / 100, 100) / self.delay(self.close, 100) <= 0.05, -1 * (self.close - self.ts_min(self.close, 100)), -1 * self.delta(self.close, 3))
    
    def alpha_025(self) -> pd.Series:
        # Alpha#25: rank(((((-1 * returns) * adv20) * vwap) * (high - close))) 
        return self.rank(-1 * self.returns * self.adv(20) * self.vwap * (self.high - self.close))
    
    def alpha_026(self) -> pd.Series:
        # Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)) 
        return -1 * self.ts_max(self.correlation(self.ts_rank(self.volume, 5), self.ts_rank(self.high, 5), 5), 3)

    def alpha_027(self) -> pd.Series:
        # Alpha#27: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1) 
        return self.condition(0.5 < self.sum(self.correlation(self.rank(self.volume), self.rank(self.vwap), 6), 2) / 2.0, -1.0, 1.0)
        
    def alpha_028(self) -> pd.Series:
        # Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close)) 
        return self.scale(self.correlation(self.adv(20), self.low, 5) + (self.high + self.low) / 2 - self.close)
        
    def alpha_029(self) -> pd.Series:
        # Alpha#29: (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5)) 
        return self.min(self.product(self.rank(self.rank(self.scale(self.log(self.sum(self.ts_min(self.rank(self.rank(-1 * self.rank(self.delta(self.close - 1.0, 5)))), 2), 1))))), 1), 5) + self.ts_rank(self.delay(-1 * self.returns, 6), 5)
        
    def alpha_030(self) -> pd.Series:
        # Alpha#30: (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
        return (1.0 - self.rank(self.sign(self.close - self.delay(self.close, 1)) + self.sign(self.delay(self.close, 1) - self.delay(self.close, 2)) + self.sign(self.delay(self.close, 2) - self.delay(self.close, 3)))) * self.sum(self.volume, 5) / self.sum(self.volume, 20)
    
    def alpha_031(self) -> pd.Series:
        # Alpha#31: ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12)))) 
        return self.rank(self.rank(self.decay_linear(self.rank(self.rank(self.delta(self.close, 10))), 10))) + self.rank(-1 * self.delta(self.close, 3)) + self.sign(self.scale(self.correlation(self.adv(20), self.low, 12)))
        
    def alpha_032(self) -> pd.Series:
        # Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230)))) 
        return self.scale(self.sum(self.close, 7) / 7 -self.close) + 20 * self.scale(self.correlation(self.vwap, self.delay(self.close, 5), 230))
        
    def alpha_033(self) -> pd.Series:
        # Alpha#33: rank((-1 * ((1 - (open / close))^1))) 
        
        return
    
    def alpha_034(self) -> pd.Series:
        # Alpha#34: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1))))) 
        return
    
    def alpha_035(self) -> pd.Series:
        # Alpha#35: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32))) 
        return
    
    def alpha_036(self) -> pd.Series:
        # Alpha#36: (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open))))) 

        return
    
    def alpha_037(self) -> pd.Series:
        # Alpha#37: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close))) 
        return
    
    def alpha_038(self) -> pd.Series:
        # Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open))) 
        return
      
    def alpha_039(self) -> pd.Series:
        # Alpha#39: ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250)))) 
        return
      
    def alpha_040(self) -> pd.Series:
        # Alpha#40: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
        return
    
    def alpha_041(self) -> pd.Series:
        # 
        return
    
    def alpha_042(self) -> pd.Series:
        # 
        return


class Alpha101Pytorch(object):
    def __init__(self, data=None):
        pass
