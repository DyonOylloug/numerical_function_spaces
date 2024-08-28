# import sys

import pytest

from numerical_function_spaces.orlicz_spaces import *


# sys.path.append('../numerical_function_spaces')  # działa z testami w pycharm
# sys.path.append('./numerical_function_spaces')  # działa z testami w terminalu pycharm
# from orlicz_spaces import *


def test_p_Amemiya_norm_big_k():
    # """Check if else is used in function."""
    x = np.array([[.1], [1]])
    assert np.isclose(p_Amemiya_norm(Orlicz_function_L_1, x=x, p_norm=1),
                      0.09999999999999999)


@pytest.mark.parametrize("p_norm, expected_result", [(1, 10), (2, 9.055385138137417), (np.inf, 9)])
def test_kappa_u_2_u_3(p_norm, expected_result):
    # Test with a 1D array
    x = np.array([[1, 2], [1, 1]])
    k = 1
    assert np.isclose(kappa(Orlicz_function_u_2_u_3, x=x, k=k, p_norm=p_norm), expected_result)


def test_p_Amemiya_norm():
    x = np.array([[1, 2], [1, 1]])
    assert np.isclose(p_Amemiya_norm(Orlicz_function_L_1_sum_L_inf, x=x, p_norm=1), 2)


def test_norm_raise_ValueError():
    """Check ValueError."""
    x = np.array([[1], [-1]])
    with pytest.raises(ValueError):
        kappa(Orlicz_function_L_1, x=x, k=1, p_norm=1)


def test_Orlicz_norm_with_stars():
    x = np.array([[1], [2]])
    np.testing.assert_allclose(Orlicz_norm_with_stars(Orlicz_function_L_1_sum_L_inf, x=x),
                               (np.float64(1.000008302431002),
                                np.float64(1.0000083024999329),
                                np.float64(1.0000083024999329)))


def test_Luxemburg_norm_with_stars():
    x = np.array([[1], [2]])
    np.testing.assert_allclose(Luxemburg_norm_with_stars(Orlicz_function_L_1_sum_L_inf, x=x),
                               (np.float64(0.666712308582787),
                                np.float64(1.4998973127189956),
                                np.float64(1.4998973127189956)))


def test_p_Amemiya_norm_with_stars():
    x = np.array([[1], [2]])
    np.testing.assert_allclose(p_Amemiya_norm_with_stars(Orlicz_function_u_2, x=x, p_norm=2),
                               (np.float64(2.000000007378271),
                                np.float64(0.7071497309222621),
                                np.float64(0.7071497309222621)))
