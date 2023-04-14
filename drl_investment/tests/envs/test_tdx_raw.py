import os
import unittest
import logging
from datetime import datetime

from drl_investment.data.tdx import unpack_data
from drl_investment.envs.tdx_raw import TDXRawEnv


LOG = logging.getLogger(__name__)


class TDXRawEnvTest(unittest.TestCase):
    '''
    reference to https://github.com/openai/gym
    example:
    import gym
    import os

    from drl_investment.data.tdx import unpack_data
    from drl_investment.envs.tdx_raw import TDXRawEnv

    file_path = os.path.join(r'E:\code\github\l1351868270\DRL-investment\drl_investment\tests', 'assets',
                                       'sh000001.day')
    data = unpack_data(file_path)
    env = gym.make("drl_investment/TDXRaw-v0", render_mode='human', data=data.to_numpy(), columns=data.columns.to_list(), )
    observation, info = env.reset(seed=42)

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()


    python -m pytest -v test_tdx_raw.py
    '''
    def setUp(self) -> None:
        self._file_path = os.path.join(os.path.dirname(__file__), 'assets',
                                       'sh000001.day')
        self._data = unpack_data(self._file_path)

        return super().setUp()

    def test_reset(self):
        env = TDXRawEnv(render_mode='human', data=self._data.to_numpy(), columns=self._data.columns.to_list())
        observation, info = env.reset(seed=42)
        LOG.info(f'observation: {observation}, info: {info}')
        env.close()
        # assert d[0] == d_0

    # def test_raw_to_data_frame(self):
    #     d = unpack_data(self._file_path)
    #     df = raw_to_data_frame(d)
    #     LOG.info(f'\ndf: \n{df}')


if __name__ == '__main__':
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
