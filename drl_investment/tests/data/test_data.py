import pytest
import os

from drl_investment.data.data import Alpha101
from drl_investment.data.tdx import unpack_data




# @pytest.mark.parametrize('data', _data)
def test_alpha101():
    file_path = os.path.join(os.path.dirname(__file__), '../', 'assets',
                                       'sh000001.day')
    _data = unpack_data(file_path)
    alpha101 = Alpha101(data=_data)
    alpha101.alphas()

