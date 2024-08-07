import numpy as np

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
