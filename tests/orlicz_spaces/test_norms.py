# import sys

import numpy as np
import pytest
from numerical_function_spaces.orlicz_spaces.norms import *

# sys.path.append('../numerical_function_spaces')  # działa z testami w pycharmie
# sys.path.append('./numerical_function_spaces')  # działa z testami w terminalu pycharma
# from orlicz_spaces import *


def Orlicz_function_u_2(u):
    return u ** 2


def Orlicz_function_u_2_u_3(u):
    return np.where(u <= 1, u ** 2, u ** 3)


def Orlicz_function_L_1_sum_L_inf(u):
    return np.where(u <= 1, 0, u - 1)


def Orlicz_function_L_1_cap_L_inf(u):
    return np.where(u <= 1, u, np.inf)


def Orlicz_function_L_1(u):
    return u


def Orlicz_function_L_inf(u):
    return np.where(u <= 1, 0, np.inf)


@pytest.mark.parametrize("p_norm, expected_result", [(1, 10), (2, 9.055385138137417), (np.inf, 9)])
def test_kappa_u_2_u_3(p_norm, expected_result):
    # Test with a 1D array
    x = np.array([[1, 2], [1, 1]])
    k = 1
    assert np.isclose(kappa(Orlicz_function_u_2_u_3, x, k, p_norm), expected_result)


def test_p_Amemiya_norm():
    x = np.array([[1, 2], [1, 1]])
    assert np.isclose(p_Amemiya_norm(Orlicz_function_L_1_sum_L_inf, x, p_norm=1), 2)

