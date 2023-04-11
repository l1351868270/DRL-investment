import os
import unittest
import logging
from datetime import datetime

from drl_investment.data.tdx import unpack_data


LOG = logging.getLogger(__name__)


class TDXTest(unittest.TestCase):
    def setUp(self) -> None:
        self._file_path = os.path.join(os.path.dirname(__file__), 'assets', 'sh000001.day')
        return super().setUp()
    
    def test_unpack_data(self):
        d = unpack_data(self._file_path)
        d_0 = [datetime.fromisoformat('1990-12-19'), '96.05', '99.98', '95.79', '99.98', '494000.0', '1260']
        LOG.info(d)
        # assert d[0] == d_0

    # def test_raw_to_data_frame(self):
    #     d = unpack_data(self._file_path)
    #     df = raw_to_data_frame(d)
    #     LOG.info(f'\ndf: \n{df}')



if __name__ == '__main__':
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
