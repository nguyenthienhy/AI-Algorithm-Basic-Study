import numpy as np


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):
    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):
    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


defines = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
           "a_u", "b_u", "c_u", "d_u",
           "e_u", "f_u", "g_u", "h_u",
           "i_u", "j_u", "k_u", "l_u",
           "m_u", "n_u", "o_u", "p_u",
           "q_u", "r_u", "s_u", "t_u",
           "u_u", "v_u", "w_u", "x_u",
           "y_u", "z_u"]

