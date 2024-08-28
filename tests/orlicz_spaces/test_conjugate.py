from numpy import inf

from numerical_function_spaces.orlicz_spaces import *


def test_conjugate_u_2():
    np.testing.assert_allclose(conjugate_function(Orlicz_function_u_2, u_max=1, du=0.1),
                               np.array([0., 0., 0.01, 0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.2]))


def test_conjugate_L_1_sum_L_inf():
    np.testing.assert_allclose(conjugate_function(Orlicz_function_L_1_sum_L_inf, u_max=2, du=0.1),
                               np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., inf, inf,
                                         inf, inf, inf, inf, inf, inf, inf]))


def test_right_side_derivative():
    np.testing.assert_allclose(right_side_derivative(Orlicz_function_u_2_u_3, u_max=2, du=0.1),
                               np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7,
                                         1.9, 3.31, 3.97, 4.69, 5.47, 6.31, 7.21, 8.17, 9.19,
                                         10.27, 11.41]))


def test_conjugate_log():
    def Orlicz_function(u):
        return 3 * (u - np.log(u + 1))

    np.testing.assert_allclose(conjugate_function(Orlicz_function, u_max=5, du=0.5),
                               np.array([0., 0., 0.21639532, 0.57944154, 1.29583687,
                                         2.87527841, inf, inf, inf, inf]))
