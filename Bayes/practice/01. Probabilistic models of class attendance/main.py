# There should be no main() in this file!!!
# Nothing should start running when you import this file somewhere.
# You may add other supporting functions to this file.
#
# Important rules:
# 1) Function pa_bc must return tensor which has dimensions (#a x #b x #c),
#    where #v is a number of different values of the variable v.
#    For input variables #v = how many input values of this variable you give to the function.
#    For output variables #v = number of all possible values of this variable.
#    Ex. for pb_a: #b = bmax-bmin+1,   #a is arbitrary.
# 2) Random variables in function names must be written in alphabetic order
#    e.g. pda_cb is an improper function name (pad_bc must be used instead)
# 3) Single dimension must be explicitly stated:
#    if you give only one value of a variable a to the function pb_a, i.e. #a=1,
#    then the function pb_a must return tensor of shape (#b, 1), not (#b,).
#
# The format of all the functions for distributions is the following:
# Inputs:
# params - dictionary with keys 'amin', 'amax', 'bmin', 'bmax', 'p1', 'p2', 'p3'
# model - model number, number from 1 to 4
# all other parameters - values of the conditions (variables a, b, c, d).
#                        Numpy vectors of size (k,), where k is an arbitrary number.
#                        For variant 3: c and d must be numpy arrays of size (k,N),
#                        where N is a number of lectures.
# Outputs:
# prob, val
# prob - probabilities for different values of the output variable with different input conditions
#        prob[i,...] = p(v=val[i]|...)
# val - support of a distribution, numpy vector of size (#v,) for variable v
#
# Example 1:
#    Function pc_ab - distribution p(c|a,b)
#    Input: a of size (k_a,) and b of size (k_b,)
#    Result: prob of size (cmax-cmin+1,k_a,k_b), val of size (cmax-cmin+1,)
#
# Example 2 (for variant 3):
#    Function pb_ad - distribution p(b|a,d_1,...,d_N)
#    Input: a of size (k_a,) and d of size (k_d,N)
#    Result: prob of size (bmax-bmin+1,k_a,k_d), val of size (bmax-bmin+1,)
#
# The format the generation function from variant 3 is the following:
# Inputs:
# N - how many points to generate
# all other inputs have the same format as earlier
# Outputs:
# d - generated values of d, numpy array of size (N,#a,#b)

# In variant 1 the following functions are required:
import numpy as np
from scipy.stats import binom, poisson


def pa(params, model):
    sup = np.arange(params["amin"], params["amax"] + 1)

    prob = np.ones_like(sup) / sup.size

    return prob, sup


def pb(params, model):
    sup = np.arange(params["bmin"], params["bmax"] + 1)

    prob = np.ones_like(sup) / sup.size

    return prob, sup


def pc(params, model):
    cmax = params["amax"] + params["bmax"]
    c_sup = np.arange(0, cmax + 1)

    a_range = np.arange(params["amin"], params["amax"] + 1)
    prob = np.sum(pc_a(a_range, params, model)[0] * pa(params, model)[0][0], axis=1)

    return prob, c_sup


def pd_c(c, params, model):
    dmax = 2 * (params["amax"] + params["bmax"])
    d_sup = np.arange(0, dmax + 1)

    k = d_sup.reshape(-1, 1) - c.reshape(1, -1)
    prob = binom.pmf(k, c, params["p3"])

    return prob, d_sup


def pd(params, model):
    dmax = 2 * (params["amax"] + params["bmax"])
    d_sup = np.arange(0, dmax + 1)

    cmax = params["amax"] + params["bmax"]
    c_range = np.arange(0, cmax + 1)

    prob = np.sum(pd_c(c_range, params, model)[0] * pc(params, model)[0].reshape(1, -1), axis=1)

    return prob, d_sup


def pc_a(a, params, model):
    cmax = params["amax"] + params["bmax"]
    c_sup = np.arange(0, cmax + 1)

    b_range = np.arange(params["bmin"], params["bmax"] + 1)
    prob = np.sum(pc_ab(a, b_range, params, model)[0] * pb(params, model)[0][0], axis=2)

    return prob, c_sup


def pc_b(b, params, model):
    cmax = params["amax"] + params["bmax"]
    c_sup = np.arange(0, cmax + 1)

    a_range = np.arange(params["amin"], params["amax"] + 1)
    prob = np.sum(pc_ab(a_range, b, params, model)[0] * pa(params, model)[0][0], axis=1)

    return prob, c_sup


def pc_d(d, params, model):
    cmax = params["amax"] + params["bmax"]
    c_sup = np.arange(0, cmax + 1)

    prob = pd_c(c_sup, params, model)[0][d] * pc(params, model)[0] / (pd(params, model)[0][d].reshape(-1, 1)[0] + 1e-15)

    return prob.transpose(1, 0), c_sup


def pc_ab(a, b, params, model):
    cmax = params["amax"] + params["bmax"]
    c_sup = np.arange(0, cmax + 1)

    if model == 1:
        prob_a = binom.pmf(c_sup.reshape(1, -1), a.reshape(-1, 1), params["p1"])
        prob_b = binom.pmf(c_sup[::-1].reshape(1, -1), b.reshape(-1, 1), params["p2"])
    else:
        # poisson
        prob_a = poisson.pmf(c_sup.reshape(1, -1), a.reshape(-1, 1) * params["p1"])
        prob_b = poisson.pmf(c_sup[::-1].reshape(1, -1), b.reshape(-1, 1) * params["p2"])

    prob = np.zeros((cmax + 1, a.size, b.size))
    for k in c_sup:
        prob[k] = prob_a[:, : k + 1] @ prob_b[:, : k + 1][::-1].T

    return prob, c_sup


def pc_abd(a, b, d, params, model):
    cmax = params["amax"] + params["bmax"]
    c_sup = np.arange(0, cmax + 1)

    pc_ab_, _ = pc_ab(a, b, params, model)
    pd_c_ = pd_c(c_sup, params, model)[0][d]

    num = pd_c_[:, :, np.newaxis, np.newaxis] * pc_ab_[np.newaxis, :, :, :]
    denom = np.tensordot(pd_c_, pc_ab_, axes=(1, 0))

    proba = num / (denom[:, np.newaxis, :, :] + 1e-15)

    return proba.transpose(1, 2, 3, 0), c_sup


def expectation(dist, sup):
    return np.dot(dist, sup)


def variance(dist, sup):
    return np.dot(dist, sup ** 2) - expectation(dist, sup) ** 2


if __name__ == "__main__":
    from time import time

    params = {"amin": 75, "amax": 90, "bmin": 500, "bmax": 600, "p1": 0.1, "p2": 0.01, "p3": 0.3}
    model = 1

    cmin = dmin = 0
    cmax = params["amax"] + params["bmax"]
    dmax = 2 * (params["amax"] + params["bmax"])

    print(f"a in [{params['amin']}:{params['amax']}]")
    print(f"b in [{params['bmin']}:{params['bmax']}]")
    print(f"c in [{cmin}:{cmax}]")
    print(f"d in [{dmin}:{dmax}]")

    a_sup = np.arange(params["amin"], params["amax"] + 1)
    b_sup = np.arange(params["bmin"], params["bmax"] + 1)
    c_sup = np.arange(cmin, cmax + 1)
    d_sup = np.arange(dmin, dmax + 1)

    start = time()
    prob, sup = pc_ab(a_sup, b_sup, params, model)
    print("p(c|ab):", prob.shape, sup.shape, f" | time spent: {time() - start}")

    start = time()
    prob, sup = pc_a(a_sup, params, model)
    print("p(c|a):", prob.shape, sup.shape, f" | time spent: {time() - start}")

    start = time()
    prob, sup = pc_b(b_sup, params, model)
    print("p(c|b):", prob.shape, sup.shape, f" | time spent: {time() - start}")

    start = time()
    prob, sup = pc(params, model)
    print("p(c):", prob.shape, sup.shape, f" | time spent: {time() - start}")

    start = time()
    prob, sup = pd_c(c_sup, params, model)
    print("p(d|c):", prob.shape, sup.shape, f" | time spent: {time() - start}")

    start = time()
    prob, sup = pd(params, model)
    print("p(d):", prob.shape, sup.shape, f" | time spent: {time() - start}")

    start = time()
    prob, sup = pc_d(d_sup, params, model)
    print("p(с|d):", prob.shape, sup.shape, f" | time spent: {time() - start}")

    start = time()
    prob, sup = pc_abd(a_sup, b_sup, d_sup, params, model)
    print("p(с|abd):", prob.shape, sup.shape, f" | time spent: {time() - start}")
