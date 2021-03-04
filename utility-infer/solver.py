#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Utility inference from choice data in a sequential decision environment
Author: Addison Bohannon
"""

import numpy as np
import scipy.linalg as sl
from numpy.random import default_rng

RNG = default_rng()

def fit(fun, sampler, prox=None, num_starts=5, tol=1e-5, max_iter=250, step_size=1e-1, return_all=False, verbose=False):
    if prox is None:
        def prox(x, t):
            return x
    nll = []
    param = []
    for start in range(num_starts):
        if verbose:
            print('Start: ' + str(start + 1) + ' / ' + str(num_starts))
        init_param = sampler()
        if verbose:
            print('Initial: ' + str(init_param))
        param_val, fun_val, _ = _fit(fun, prox, init_param, tol=tol, max_iter=max_iter, step_size=step_size, verbose=verbose)
        nll.append(fun_val)
        param.append(param_val)
    if not return_all:
        opt = np.nanargmax(nll)
        nll = nll[opt]
        param = param[opt]
    return param, nll


def _fit(fun, prox, init_param, tol=1e-5, max_iter=250, step_size=1e-1, verbose=False):
    param, old_param, best_param = np.copy(init_param), np.copy(init_param), np.copy(init_param)
    best_fun_val = np.inf
    res = []
    for curr_iter in range(max_iter):
        fun_val, grad_val = fun(param)
        if fun_val < best_fun_val:
            best_fun_val = np.copy(fun_val)
            best_param = np.copy(param)
        param -= step_size * grad_val / np.sqrt(curr_iter + 1)
        param = prox(param, step_size)
        res.append(sl.norm(param - old_param))
        if verbose:
            print('Iteration: ' + str(curr_iter) + ', Estimate: ' + str(param) + ', Value: ' + str(fun_val) + ', Residual: ' + str(res[-1]))
        if curr_iter > 0 and (res[curr_iter] < tol * res[0] or res[curr_iter] < tol):
            break
        else:
            old_param = np.copy(param)
    if fun_val < best_fun_val:
        best_fun_val = np.copy(fun_val)
        best_param = np.copy(param)
    return best_param, best_fun_val, res
