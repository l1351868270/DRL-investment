import abc
import logging
import pandas as pd
import numpy as np
from typing import Union, Callable

LOG = logging.getLogger(__name__)


class FunctionNotImplement(BaseException):
    def __init__(self, f: Callable, *args):
        super().__init__(args)
        self.f = f

    def __str__(self):
        return f'The alpha function {self.f.__qualname__} has not implement'


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
        self.adv20: pd.Series = self.adv(self.amount, 20)

        self.invalid = ['alpha_048', 'alpha_056', 'alpha_058', 'alpha_059', 'alpha_063', 'alpha_067',
                            'alpha_069', 'alpha_070', 'alpha_076', 'alpha_078', 'alpha_079', 'alpha_080',
                            'alpha_082', 'alpha_085', 'alpha_087', 'alpha_089', 'alpha_090', 'alpha_091',
                            'alpha_093', 'alpha_094', 'alpha_097', 'alpha_100',
                            ]

    def alphas(self, columns: list[str] = None, include_raw: bool = False) -> pd.DataFrame:
        ss = []
        for i in range(1, 102):
            name = f'alpha_{i:0>3d}'
            if name not in self.invalid:
                ss.append(getattr(self, name)())
        return pd.DataFrame(list(zip(ss)))

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

    # def min(self, x: pd.Series, d: int) -> pd.Series:
    #     # min(x, d) = ts_min(x, d)
    #     return self.ts_min(x, d)

    def min(self, x: pd.Series, y: pd.Series) -> pd.Series:
        # min(x, d) = ts_min(x, d)
        return self.condition(x < y, x, y)

    # def max(self, x: pd.Series, d: int) -> pd.Series:
    #     # max(x, d) = ts_max(x, d)
    #     return self.ts_max(x, d)
    
    def max(self, x: pd.Series, y: pd.Series) -> pd.Series:
        # max(x, d) = ts_max(x, d) 貌似错了
        return self.condition(x < y, y, x)
    
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

    def pow(self, x: pd.Series, a):
        return x.pow(a)

    def condition(self, con: pd.Series, x: Union[pd.Series, int, float], y: Union[pd.Series, int, float]) -> pd.Series:
        if (not isinstance(x, pd.Series)) and (not isinstance(y, pd.Series)):
            inner = pd.Series([np.float32(y) for _ in range(len(con))], self.open.index)
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
        return self.rank(self.ts_argmax(self.signedpower(self.condition(self.returns < 0.0, self.stddev(self.returns, 20), self.close), 2), 5)) - 0.5

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
        return self.condition(self.adv20 < self.volume, -1 * self.ts_rank(self.abs(self.delta(self.close, 7)), 60), -1 * 1)

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
        return -1 * self.rank(self.ts_rank(self.close, 10)) * self.rank(self.delta(self.delta(self.close, 1), 1)) * self.rank(self.ts_rank(self.volume / self.adv20, 5))

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
        return self.condition(self.sum(self.close, 8) / 8 + self.stddev(self.close, 8) < self.sum(self.close, 2) / 2, -1, self.condition(self.sum(self.close, 2) / 2 < self.sum(self.close, 8) - self.stddev(self.close, 8), 1, self.condition(self.volume / self.adv20 >= 1, 1, -1)))

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
        return self.rank(-1 * self.returns * self.adv20 * self.vwap * (self.high - self.close))

    def alpha_026(self) -> pd.Series:
        # Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
        return -1 * self.ts_max(self.correlation(self.ts_rank(self.volume, 5), self.ts_rank(self.high, 5), 5), 3)

    def alpha_027(self) -> pd.Series:
        # Alpha#27: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
        return self.condition(0.5 < self.sum(self.correlation(self.rank(self.volume), self.rank(self.vwap), 6), 2) / 2.0, -1.0, 1.0)

    def alpha_028(self) -> pd.Series:
        # Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
        return self.scale(self.correlation(self.adv20, self.low, 5) + (self.high + self.low) / 2 - self.close)

    def alpha_029(self) -> pd.Series:
        # Alpha#29: (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
        return self.min(self.product(self.rank(self.rank(self.scale(self.log(self.sum(self.ts_min(self.rank(self.rank(-1 * self.rank(self.delta(self.close - 1.0, 5)))), 2), 1))))), 1), 5) + self.ts_rank(self.delay(-1 * self.returns, 6), 5)

    def alpha_030(self) -> pd.Series:
        # Alpha#30: (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
        return (1.0 - self.rank(self.sign(self.close - self.delay(self.close, 1)) + self.sign(self.delay(self.close, 1) - self.delay(self.close, 2)) + self.sign(self.delay(self.close, 2) - self.delay(self.close, 3)))) * self.sum(self.volume, 5) / self.sum(self.volume, 20)

    def alpha_031(self) -> pd.Series:
        # Alpha#31: ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
        return self.rank(self.rank(self.decay_linear(self.rank(self.rank(self.delta(self.close, 10))), 10))) + self.rank(-1 * self.delta(self.close, 3)) + self.sign(self.scale(self.correlation(self.adv20, self.low, 12)))

    def alpha_032(self) -> pd.Series:
        # Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))
        return self.scale(self.sum(self.close, 7) / 7 -self.close) + 20 * self.scale(self.correlation(self.vwap, self.delay(self.close, 5), 230))

    def alpha_033(self) -> pd.Series:
        # Alpha#33: rank((-1 * ((1 - (open / close))^1)))
        self.rank(-1 * (1 - self.open / self.close).pow(1))
        return

    def alpha_034(self) -> pd.Series:
        # Alpha#34: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
        return self.rank(1 - self.rank(self.stddev(self.returns, 2) / self.stddev(self.returns, 5)) + (1 - self.rank(self.delta(self.close, 1))))

    def alpha_035(self) -> pd.Series:
        # Alpha#35: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))
        return self.ts_rank(self.volume, 32) * (1 - self.ts_rank((self.close + self.high - self.low), 16)) * (1 - self.ts_rank(self.returns, 32))

    def alpha_036(self) -> pd.Series:
        # Alpha#36: (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
        return 2.21 * self.rank(self.correlation(self.close - self.open, self.delay(self.volume, 1), 15)) + 0.7 * self.rank(self.open - self.close) + 0.73 * self.rank(self.ts_rank(self.delay(-1 * self.returns, 6), 5)) + self.rank(self.abs(self.correlation(self.vwap, self.adv20, 6))) + 0.6 * self.rank((self.sum(self.close, 200) / 200 - self.open) * (self.close - self.open))

    def alpha_037(self) -> pd.Series:
        # Alpha#37: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
        return self.rank(self.correlation(self.delay(self.open - self.close, 1), self.close, 200)) + self.rank(self.open - self.close)

    def alpha_038(self) -> pd.Series:
        # Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
        return -1 * self.rank(self.ts_rank(self.close, 10)) * self.rank(self.close / self.open)

    def alpha_039(self) -> pd.Series:
        # Alpha#39: ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))
        return -1 * self.rank(self.delta(self.close, 7) * (1 - self.rank(self.decay_linear(self.volume / self.adv20, 9)))) * (1 + self.rank(self.sum(self.returns, 250)))

    def alpha_040(self) -> pd.Series:
        # Alpha#40: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
        return -1 * self.rank(self.stddev(self.high, 10)) * self.correlation(self.high, self.volume, 10)

    def alpha_041(self) -> pd.Series:
        # Alpha#41: (((high * low)^0.5) - vwap)
        return self.pow(self.high * self.low, 0.5) - self.vwap

    def alpha_042(self) -> pd.Series:
        # Alpha#42: (rank((vwap - close)) / rank((vwap + close)))
        return self.rank(self.vwap - self.close) / self.rank(self.vwap + self.close)

    def alpha_043(self) -> pd.Series:
        # Alpha#43: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
        return self.ts_rank(self.volume / self.adv20, 20) * self.ts_rank(-1 * self.delta(self.close, 7), 8)

    def alpha_044(self) -> pd.Series:
        # Alpha#44: (-1 * correlation(high, rank(volume), 5))
        return -1 * self.correlation(self.high, self.rank(self.volume), 5)

    def alpha_045(self) -> pd.Series:
        # Alpha#45: (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))
        return self.rank(self.sum(self.delay(self.close, 5), 20) / 20) * self.correlation(self.close, self.volume, 2) * self.rank(self.correlation(self.sum(self.close, 5), self.sum(self.close, 20), 2))

    def alpha_046(self) -> pd.Series:
        # Alpha#46: ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))
        inner = ((self.delay(self.close, 20) - self.delay(self.close, 10)) / 10 - (self.delay(self.close, 10) - self.close) / 10 )
        return self.condition(0.25 < inner, -1 * 1, self.condition(inner < 0, 1, -1 * 1 * self.close - self.delay(self.close, 1)))

    def alpha_047(self) -> pd.Series:
        # Alpha#47: ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))
        return (self.rank(1 / self.close) * self.volume) / self.adv20 * (self.high * self.rank(self.high - self.close) / (self.sum(self.high, 5) / 5)) - self.rank(self.vwap - self.delay(self.vwap, 5))

    def alpha_048(self) -> pd.Series:
        # Alpha#48: (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))
        raise FunctionNotImplement(self.alpha_048)

    def alpha_049(self) -> pd.Series:
        # Alpha#49: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        return self.condition((self.delay(self.close, 20) - self.delay(self.close , 10) / 10 - (self.delay(self.close , 10) - self.close) / 10) < -1 * 0.1, 1, -1 * 1 * (self.close - self.delay(self.close , 1)))

    def alpha_050(self) -> pd.Series:
        # Alpha#50: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
        return self.ts_max(self.rank(self.correlation(self.rank(self.volume), self.rank(self.vwap), 5)), 5)

    def alpha_051(self) -> pd.Series:
        # Alpha#51: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        return self.condition((self.delay(self.close, 20) - self.delay(self.close , 10)) / 10 - (self.delay(self.close, 10) - self.close) / 10 < -1 * 0.05, 1, -1 * 1 * (self.close - self.delay(self.close, 1)))

    def alpha_052(self) -> pd.Series:
        # Alpha#52: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))
        return (-1 * self.ts_min(self.low, 5) + self.delay(self.ts_min(self.low, 5), 5)) * self.rank((self.sum(self.returns, 240) - self.sum(self.returns, 20)) / 220) * self.ts_rank(self.volume, 5)

    def alpha_053(self) -> pd.Series:
        # Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
        return -1 * self.delta(((self.close - self.low) - (self.high - self.close)) / (self.close - self.low), 9)

    def alpha_054(self) -> pd.Series:
        # Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
        return (-1 * (self.low - self.close) * self.pow(self.open, 5)) / ((self.low - self.high) * self.pow(self.close, 5))

    def alpha_055(self) -> pd.Series:
        # Alpha#55: (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))
        return -1 * self.correlation(self.rank((self.close - self.ts_min(self.low, 12)) / (self.ts_max(self.high, 12) - self.ts_min(self.low, 12))), self.rank(self.volume), 6)

    def alpha_056(self) -> pd.Series:
        # Alpha#56: (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
        # return 0 - 1 * self.rank(self.sum(self.returns, 10) / self.sum(self.sum(self.returns, 2), 3)) * self.rank(self.returns * self.cap)
        raise FunctionNotImplement(self.alpha_056)

    def alpha_057(self) -> pd.Series:
        # Alpha#57: (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
        return 0 - 1 * (self.close - self.vwap) / self.decay_linear(self.rank(self.ts_argmax(self.close, 30)), 2)

    def alpha_058(self) -> pd.Series:
        # Alpha#58: (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))
        raise FunctionNotImplement(self.alpha_058)

    def alpha_059(self) -> pd.Series:
        # Alpha#59: (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))
        raise FunctionNotImplement(self.alpha_059)

    def alpha_060(self) -> pd.Series:
        # Alpha#60: (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))
        return 0 - 1 * (2 * self.scale(self.rank((((self.close - self.low) - (self.high - self.close)) / (self.high - self.low)) * self.volume)) - self.scale(self.rank(self.ts_argmax(self.close, 10))))

    def alpha_061(self) -> pd.Series:
        # Alpha#61: (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
        return self.condition(self.rank(self.vwap - self.ts_min(self.vwap, 16)) < self.rank(self.correlation(self.vwap, self.adv(self.amount, 180), 18)), 1.0, 0.0)

    def alpha_062(self) -> pd.Series:
        # Alpha#62: ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
        return self.condition(self.rank(self.correlation(self.vwap, self.sum(self.adv20, 22), 10)) < self.condition(self.rank(self.open) + self.rank(self.open) < (self.rank((self.high + self.low) / 2) + self.rank(self.high)), 1.0, 0.0), 1.0, 0.0) * -1

    def alpha_063(self) -> pd.Series:
        # Alpha#63: ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)
        raise FunctionNotImplement(self.alpha_063)

    def alpha_064(self) -> pd.Series:
        # Alpha#64: ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)
        return self.condition(self.rank(self.correlation(self.sum(self.open * 0.178404 + self.low * (1 - 0.178404), 13), self.sum(self.adv(self.amount, 120), 13), 17)) < self.rank(self.delta((((self.high + self.low) / 2) * 0.178404) + self.vwap * (1 - 0.178404), 4)), 1.0, 0.0) * -1

    def alpha_065(self) -> pd.Series:
        # Alpha#65: ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
        return self.condition(self.rank(self.correlation(self.open * 0.00817205 + self.vwap * (1 - 0.00817205), self.sum(self.adv(self.amount, 60), 9), 6)) < self.rank(self.open - self.ts_min(self.open, 14)), 1.0, 0.0) * -1

    def alpha_066(self) -> pd.Series:
        # Alpha#66: ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
        return (self.rank(self.decay_linear(self.delta(self.vwap, 4), 7)) + self.ts_rank(self.decay_linear((self.low * 0.96633 + self.low * (1 - 0.96633) - self.vwap) / (self.open - (self.high + self.low) / 2), 11), 7)) * -1

    def alpha_067(self) -> pd.Series:
        # Alpha#67: ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)
        raise FunctionNotImplement(self.alpha_067)

    def alpha_068(self) -> pd.Series:
        # Alpha#68: ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
        return self.condition(self.ts_rank(self.correlation(self.rank(self.high), self.rank(self.adv(self.amount, 15)), 9), 14) < self.rank(self.delta(self.close * 0.518371 + self.low * (1 - 0.518371), 1)), 1.0, 0.0) * -1

    def alpha_069(self) -> pd.Series:
        # Alpha#69: ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)
        raise FunctionNotImplement(self.alpha_069)

    def alpha_070(self) -> pd.Series:
        # Alpha#70: ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)
        raise FunctionNotImplement(self.alpha_070)

    def alpha_071(self) -> pd.Series:
        # Alpha#71: max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))
        return self.max(self.ts_rank(self.decay_linear(self.correlation(self.ts_rank(self.close , 3), self.ts_rank(self.adv(self.amount, 180), 12), 18), 4), 15), self.ts_rank(self.decay_linear(self.pow(self.rank((self.low + self.open) - (self.vwap + self.vwap)), 2), 16), 4))

    def alpha_072(self) -> pd.Series:
        # Alpha#72: (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))
        return self.rank(self.decay_linear(self.correlation((self.high + self.low) / 2, self.adv(self.amount, 40), 9), 10)) / self.rank(self.decay_linear(self.correlation(self.ts_rank(self.vwap, 4), self.ts_rank(self.volume, 19), 7), 3))

    def alpha_073(self) -> pd.Series:
        # Alpha#73: (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
        return self.max(self.decay_linear(self.delta(self.vwap, 5), 3), self.ts_rank(self.decay_linear((self.delta(self.open * 0.147155 + self.low * (1 - 0.147155), 2) / (self.open * 0.147155 + self.low * (1 - 0.147155))) * -1, 3), 17)) * -1

    def alpha_074(self) -> pd.Series:
        # Alpha#74: ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)
        return self.condition(self.rank(self.correlation(self.close, self.sum(self.adv(self.amount, 30), 37), 15)) < self.rank(self.correlation(self.rank(self.high * 0.0261661 + self.vwap * (1 - 0.0261661)), self.rank(self.volume), 11)), 1.0, 0.0) * -1

    def alpha_075(self) -> pd.Series:
        # Alpha#75: (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))
        return self.condition(self.rank(self.correlation(self.vwap, self.volume, 4)) < self.rank(self.correlation(self.rank(self.low), self.rank(self.adv(self.amount, 50)), 12)), 1.0, 0.0)

    def alpha_076(self) -> pd.Series:
        # Alpha#76: (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1)
        raise FunctionNotImplement(self.alpha_076)

    def alpha_077(self) -> pd.Series:
        # Alpha#77: min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
        return self.min(self.rank(self.decay_linear(((self.high + self.low) / 2 + self.high) - (self.vwap + self.high), 20)), self.rank(self.decay_linear(self.correlation((self.high + self.low) / 2, self.adv(self.amount, 40), 3), 6)))

    def alpha_078(self) -> pd.Series:
        # Alpha#78: (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
        self.rank(self.correlation(self.sum(self.low * 0.352233 + self.vwap(1 - 0.352233), 20), self.sum(self.adv(self.amount, 40), 20), 7))
        self.rank(self.correlation(self.rank(self.vwap), self.rank(self.volume), 6))
        raise FunctionNotImplement(self.alpha_078)

    def alpha_079(self) -> pd.Series:
        # Alpha#79: (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))), IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644)))
        raise FunctionNotImplement(self.alpha_079)

    def alpha_080(self) -> pd.Series:
        # Alpha#80: ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))), IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)
        raise FunctionNotImplement(self.alpha_080)

    def alpha_081(self) -> pd.Series:
        # Alpha#81: ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
        return self.condition(self.rank(self.log(self.product(self.rank(self.pow(self.rank(self.correlation(self.vwap, self.sum(self.adv(self.amount, 10), 50), 8)), 4)), 15))) < self.rank(self.correlation(self.rank(self.vwap), self.rank(self.volume), 5)), 1.0, 0.0) * -1

    def alpha_082(self) -> pd.Series:
        # Alpha#82: (min(rank(decay_linear(delta(open, 1.46063), 14.8717)), Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)
        raise FunctionNotImplement(self.alpha_082)

    def alpha_083(self) -> pd.Series:
        # Alpha#83: ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))
        return (self.rank(self.delay((self.high - self.low) / (self.sum(self.close, 5) / 5), 2)) * self.rank(self.rank(self.volume))) / (((self.high - self.low) / (self.sum(self.close, 5) / 5)) / (self.vwap - self.close))

    def alpha_084(self) -> pd.Series:
        # Alpha#84: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))
        return self.signedpower(self.ts_rank(self.vwap - self.ts_max(self.vwap, 15), 21), self.delta(self.close, 5))

    def alpha_085(self) -> pd.Series:
        # Alpha#85: (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))
        self.rank(self.correlation(self.high * 0.876703 + self.close * (1 - 0.876703), self.adv(self.amount, 30), 10))
        self.rank(self.correlation(self.ts_rank((self.high + self.low) / 2, 4), self.ts_rank(self.volume, 10), 7))
        raise FunctionNotImplement(self.alpha_085)

    def alpha_086(self) -> pd.Series:
        # Alpha#86: ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open + close) - (vwap + open)))) * -1)
        return self.condition(self.ts_rank(self.correlation(self.close, self.sum(self.adv20, 15), 6), 20) < self.rank((self.open + self.close) - (self.vwap + self.open)), 1.0, 0.0) * -1

    def alpha_087(self) -> pd.Series:
        # Alpha#87: (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
        raise FunctionNotImplement(self.alpha_087)

    def alpha_088(self) -> pd.Series:
        # Alpha#88: min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))
        return self.min(self.rank(self.decay_linear((self.rank(self.open) + self.rank(self.low)) - (self.rank(self.high) + self.rank(self.close)), 8)), self.ts_rank(self.decay_linear(self.correlation(self.ts_rank(self.close, 8), self.ts_rank(self.adv(self.amount, 60), 21), 8), 7), 3))

    def alpha_089(self) -> pd.Series:
        # Alpha#89: (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012))
        raise FunctionNotImplement(self.alpha_089)

    def alpha_090(self) -> pd.Series:
        # Alpha#90: ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry), low, 5.38375), 3.21856)) * -1)
        raise FunctionNotImplement(self.alpha_090)

    def alpha_091(self) -> pd.Series:
        # Alpha#91: ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)
        raise FunctionNotImplement(self.alpha_091)

    def alpha_092(self) -> pd.Series:
        # Alpha#92: min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))
        return self.min(self.ts_rank(self.decay_linear(self.condition(((self.high + self.low) / 2 + self.close) < (self.low + self.open), 1.0, 0.0), 15), 19), self.ts_rank(self.decay_linear(self.correlation(self.rank(self.low), self.rank(self.adv(self.amount, 30)), 8), 7), 7))

    def alpha_093(self) -> pd.Series:
        # Alpha#93: (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))
        raise FunctionNotImplement(self.alpha_093)

    def alpha_094(self) -> pd.Series:
        # Alpha#94: ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
        self.rank(self.vwap - self.ts_min(self.vwap, 12))
        self.ts_rank(self.correlation(self.ts_rank(self.vwap, 20), self.ts_rank(self.adv(self.amount, 60), 4), 18), 3)
        raise FunctionNotImplement(self.alpha_094)

    def alpha_095(self) -> pd.Series:
        # Alpha#95: (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
        return self.condition(self.rank(self.open - self.ts_min(self.open, 12)) < self.ts_rank(self.pow(self.rank(self.correlation(self.sum((self.high + self.low) / 2, 19), self.sum(self.adv(self.amount, 40), 19), 13)), 5), 12), 1.0, 0.0)

    def alpha_096(self) -> pd.Series:
        # Alpha#96: (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
        return self.max(self.ts_rank(self.decay_linear(self.correlation(self.rank(self.vwap), self.rank(self.volume), 4), 4), 8),    self.ts_rank(self.decay_linear(self.ts_argmax(self.correlation(self.ts_rank(self.close, 7), self.ts_rank(self.adv(self.amount, 60), 4), 4), 13), 14), 13)) * -1

    def alpha_097(self) -> pd.Series:
        # Alpha#97: ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
        raise FunctionNotImplement(self.alpha_097)

    def alpha_098(self) -> pd.Series:
        # Alpha#98: (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))
        return self.rank(self.decay_linear(self.correlation(self.vwap, self.sum(self.adv(self.amount, 5), 26), 5), 7)) - self.rank(self.decay_linear(self.ts_rank(self.ts_argmin(self.correlation(self.rank(self.open), self.rank(self.adv(self.amount, 15)), 21), 9), 7), 8))

    def alpha_099(self) -> pd.Series:
        # Alpha#99: ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1)
        return self.condition(self.rank(self.correlation(self.sum((self.high + self.low) / 2, 20), self.sum(self.adv(self.amount, 60), 20), 9)) < self.rank(self.correlation(self.low, self.volume, 6)), 1.0, 0.0) * -1

    def alpha_100(self) -> pd.Series:
        # Alpha#100: (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) - scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20))))
        raise FunctionNotImplement(self.alpha_100)

    def alpha_101(self) -> pd.Series:
        # Alpha#101: ((close - open) / ((high - low) + .001))
        return (self.close - self.open) / ((self.high - self.low) + 0.001)


class Alpha101Pytorch(object):
    def __init__(self, data=None):
        pass
