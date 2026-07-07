"""Numba-accelerated per-trial negative log-likelihood for the resource-allocation
models. Each kernel runs the model forward through the trial sequence, updating its
value and precision estimates online, and accumulates the choice NLL.

Standard-normal cdf/pdf are evaluated with math.erfc / a direct pdf, which match
scipy to machine precision but run ~60x faster in the per-trial loop.

model_id: 0=DRA, 1=FreqRA, 2=StakesRA, 3=EqualRA
params x = [lr_v, lr_s, lmda, sigma_0];  sigma_base fixed at 5.0

`n0` supports warm-start cross-validation: the model state updates over every
trial, but the NLL is accumulated only from trial index `n0` onward.
"""

import math

import numpy as np
from numba import njit

SQRT1_2 = 0.7071067811865476
INV_SQRT_2PI = 0.3989422804014327
PROB_FLOOR = 0.001  # clip choice probabilities away from 0 before taking the log
SIGMA_FLOOR = 0.01  # lower clip on the precision (sigma)
N_BACK = 25  # window for the running-average precision in the scalar models

_s3 = math.sqrt(3.0)
_den = 2.0 * _s3 + 2.0
NORM_FREQ = np.array([_s3, _s3, 1.0, 1.0]) * 4.0 / _den
NORM_STAKES = np.array([_s3, 1.0, _s3, 1.0]) * 4.0 / _den
NORM_EQUAL = np.array([1.0, 1.0, 1.0, 1.0])
NORMS = {1: NORM_FREQ, 2: NORM_STAKES, 3: NORM_EQUAL}


@njit(cache=True, fastmath=False)
def _ndtr(x):
    return 0.5 * math.erfc(-x * SQRT1_2)


@njit(cache=True, fastmath=False)
def _npdf(x):
    return INV_SQRT_2PI * math.exp(-0.5 * x * x)


@njit(cache=True, fastmath=False)
def nll_dra(p, sm, price, reward, action, n0=0):
    lr_v, lr_s, lmda, sigma_0 = p[0], p[1], p[2], p[3]
    sb = 5.0
    v = np.zeros(4)
    sigma = np.empty(4)
    for k in range(4):
        sigma[k] = sigma_0
    nll = 0.0
    n = sm.shape[0]
    gc = np.empty(4)
    for i in range(n):
        s = sm[i]
        pr = price[i]
        rw = reward[i]
        a = action[i]
        p_no = _ndtr((pr - v[s]) / sigma[s])
        prob = (1.0 - p_no) if a == 0 else p_no
        if prob < PROB_FLOOR:
            prob = PROB_FLOOR
        elif prob > 1.0:
            prob = 1.0
        if i >= n0:
            nll -= math.log(prob)
        for k in range(4):
            gc[k] = sigma[k] / (sb * sb) - 1.0 / sigma[k]
        grad_reward = 0.0
        if a == 0:
            v[s] += lr_v * (rw - v[s])
            x = (pr - v[s]) / sigma[s]
            grad_reward = _npdf(x) / (1.0 - _ndtr(x) + 1e-4) * x / sigma[s]
            grad_reward *= rw
        sigma[s] += lr_s * grad_reward
        for k in range(4):
            sigma[k] -= lr_s * lmda * gc[k]
            if sigma[k] < SIGMA_FLOOR:
                sigma[k] = SIGMA_FLOOR
            elif sigma[k] > sb:
                sigma[k] = sb
    return nll


@njit(cache=True, fastmath=False)
def nll_other(p, sm, price, reward, action, norm, n0=0):
    lr_v, lr_s, lmda, sigma_0 = p[0], p[1], p[2], p[3]
    sb = 5.0
    v = np.zeros(4)
    sigma = np.empty(4)
    for k in range(4):
        sigma[k] = sigma_0 / norm[k]
    sigma_scalar = sigma_0
    n = sm.shape[0]
    hist = np.empty(n)
    nll = 0.0
    for i in range(n):
        s = sm[i]
        pr = price[i]
        rw = reward[i]
        a = action[i]
        p_no = _ndtr((pr - v[s]) / sigma[s])
        prob = (1.0 - p_no) if a == 0 else p_no
        if prob < PROB_FLOOR:
            prob = PROB_FLOOR
        elif prob > 1.0:
            prob = 1.0
        if i >= n0:
            nll -= math.log(prob)
        grad_cost = 0.0
        for k in range(4):
            grad_cost += sigma[k] / (sb * sb) - 1.0 / sigma[k]
        grad_reward = 0.0
        if a == 0:
            v[s] += lr_v * (rw - v[s])
            x = (pr - v[s]) / sigma[s]
            grad_reward = _npdf(x) / (1.0 - _ndtr(x) + 1e-4) * x / sigma[s]
            grad_reward *= rw / norm[s]
        sigma_scalar += lr_s * (grad_reward - lmda * grad_cost)
        hist[i] = sigma_scalar
        count = i + 1
        start = count - N_BACK if count > N_BACK else 0
        m = 0.0
        for j in range(start, count):
            m += hist[j]
        m /= count - start
        for k in range(4):
            sigma[k] = m / norm[k]
        if sigma_scalar < SIGMA_FLOOR:
            sigma_scalar = SIGMA_FLOOR
        elif sigma_scalar > sb:
            sigma_scalar = sb
        for k in range(4):
            if sigma[k] < SIGMA_FLOOR:
                sigma[k] = SIGMA_FLOOR
            elif sigma[k] > sb:
                sigma[k] = sb
    return nll


def make_nll(model_id, sm, price, reward, action, n0=0):
    """Return a scalar objective f(x) -> NLL for one model and one trial block.
    With n0>0 the model state is warmed up over trials [0:n0] but only trials
    [n0:] contribute to the returned NLL (warm-start cross-validation)."""
    if model_id == 0:

        def f(x):
            return nll_dra(np.asarray(x, np.float64), sm, price, reward, action, n0)

    else:
        norm = NORMS[model_id]

        def f(x):
            return nll_other(
                np.asarray(x, np.float64), sm, price, reward, action, norm, n0
            )

    return f
