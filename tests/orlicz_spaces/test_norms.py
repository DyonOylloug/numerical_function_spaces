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


def test_kappa_raise_ValueError():
    """Check ValueError."""
    x = np.array([[1], [-1]])
    with pytest.raises(ValueError):
        kappa(Orlicz_function_L_1, x=x, k=1, p_norm=1)


def test_norm_raise_ValueError():
    """Check ValueError."""
    x = np.array([[1], [-1]])
    with pytest.raises(ValueError):
        p_Amemiya_norm(Orlicz_function_L_1, x=x, p_norm=1)


def test_norm_with_stars_raise_ValueError():
    """Check ValueError."""
    x = np.array([[dc.Decimal(1)], [dc.Decimal(-1)]])
    with pytest.raises(ValueError):
        p_Amemiya_norm_with_stars(Orlicz_function_L_1, x=x, p_norm=1)


def test_norm_decimal_raise_ValueError():
    """Check ValueError."""
    x = np.array([[dc.Decimal(1)], [dc.Decimal(-1)]])
    with pytest.raises(ValueError):
        p_Amemiya_norm_with_stars_by_decimal(Orlicz_function_L_1, x=x, p_norm=1)


def test_norm_zeros():
    x = np.array([[0], [2]])
    assert (
            p_Amemiya_norm(Orlicz_function_L_1_sum_L_inf, x=x, p_norm=1)[0] == 0
            and np.isnan(p_Amemiya_norm(Orlicz_function_L_1_sum_L_inf, x=x, p_norm=1)[1])
            and np.isnan(p_Amemiya_norm(Orlicz_function_L_1_sum_L_inf, x=x, p_norm=1)[2])
    )


def test_norm_with_stars_zeros():
    x = np.array([[0], [2]])
    assert (
            p_Amemiya_norm_with_stars(Orlicz_function_L_1_sum_L_inf, x=x, p_norm=1)[0] == 0
            and np.isnan(p_Amemiya_norm_with_stars(Orlicz_function_L_1_sum_L_inf, x=x, p_norm=1)[1])
            and np.isnan(p_Amemiya_norm_with_stars(Orlicz_function_L_1_sum_L_inf, x=x, p_norm=1)[2])
    )


def test_norm_with_stars_by_decimal_zeros():
    x = np.array([[dc.Decimal(0)], [dc.Decimal(2)]])
    assert (
            p_Amemiya_norm_with_stars_by_decimal(Orlicz_function_L_1_sum_L_inf, x=x, p_norm=1)[0] == 0
            and np.isnan(p_Amemiya_norm_with_stars_by_decimal(Orlicz_function_L_1_sum_L_inf, x=x, p_norm=1)[1])
            and np.isnan(p_Amemiya_norm_with_stars_by_decimal(Orlicz_function_L_1_sum_L_inf, x=x, p_norm=1)[2])
    )


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


def test_norm_with_dk():
    x = np.array([[1], [1]])
    np.testing.assert_allclose(p_Amemiya_norm_with_stars(Orlicz_function_L_1_cap_L_inf,
                                                         x=x,
                                                         p_norm=1,
                                                         dk=0.1),
                               (np.float64(2.0001109956018714),
                                np.float64(0.9998890167167849),
                                np.float64(0.9998890167167849)))


def test_norm_by_decimal_with_dk():
    x = np.array([[dc.Decimal(1)], [dc.Decimal(1)]])
    assert p_Amemiya_norm_with_stars_by_decimal(Orlicz_function_u_2,
                                                x=x,
                                                p_norm=dc.Decimal(1),
                                                dk=dc.Decimal('0.1')) == (dc.Decimal('2.000000000099999000009999900'),
                                                                          dc.Decimal('1.00001'),
                                                                          dc.Decimal('1.00001'))


def test_norm_by_decimal_inf():
    x = np.array([[dc.Decimal(1)], [dc.Decimal(1)]])
    assert p_Amemiya_norm_with_stars_by_decimal(Orlicz_function_u_2,
                                                x=x,
                                                p_norm=dc.Decimal(np.inf)) == (
               dc.Decimal('1.000009900000000000000000000'),
               dc.Decimal('1.00000990'),
               dc.Decimal('1.00000990'))


def test_norm_by_decimal_2():
    x = np.array([[dc.Decimal(1)], [dc.Decimal(1)]])
    assert p_Amemiya_norm_with_stars_by_decimal(Orlicz_function_u_2,
                                                x=x,
                                                p_norm=dc.Decimal(2)) == (dc.Decimal('1.414213562511700747850059004'),
                                                                          dc.Decimal('1.00000990'),
                                                                          dc.Decimal('1.00000990'))


def test_norm_extend_domain_k():
    x = np.array([[1], [1]])
    np.testing.assert_allclose(p_Amemiya_norm_with_stars(Orlicz_function_L_inf,
                                                         x=0.001 * x,
                                                         p_norm=1),
                               (np.float64(0.0010000192259169113),
                                np.float64(999.9807744527175),
                                np.float64(999.9807744527175)))


def test_norm_k_star_out_of_domain():
    def Orlicz_function(u):
        Phi = np.zeros(len(u))
        for i in range(len(u)):
            if u[i] <= 1:
                Phi[i] = u[i] ** 2
            elif u[i] <= 2:
                Phi[i] = 2 * u[i] - 1
            else:
                Phi[i] = (u[i] - 1) ** 2 + 2
        return Phi

    x = np.array([[1], [1]])
    np.testing.assert_allclose(p_Amemiya_norm_with_stars(Orlicz_function,
                                                         x=x,
                                                         p_norm=1,
                                                         k_min=1.1,
                                                         k_max=1.9),
                               (np.float64(1.9999999999999998),
                                np.float64(1.1),
                                np.float64(1.899199999999912)))


def test_norm_k_star_by_decimal_out_of_domain():
    def Orlicz_function(u):
        return np.where(u <= 1, u ** 2, np.where(u <= 2, 2 * u - 1, (u - 1) ** 2 + 2))

    x = np.array([[1], [1]])
    assert p_Amemiya_norm_with_stars_by_decimal(Orlicz_function,
                                                x=x,
                                                p_norm=dc.Decimal(1),
                                                k_min=dc.Decimal('1.1'),
                                                k_max=dc.Decimal('1.9')
                                                ) == (
           dc.Decimal('2.000000000000000000000000000'), dc.Decimal('1.1'), dc.Decimal('1.8992'))
