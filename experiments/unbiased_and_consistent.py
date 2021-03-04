#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Utility inference from choice data in a sequential decision environment
Author: Addison Bohannon
"""

import numpy as np
import matplotlib.pyplot as plt
from solver import fit
from agent import SoftmaxAgent
from utility import uniform_sampler, proj, nll_mdp
from inventory_control import InventoryControlEnvironment, InventoryControlRewardWithRisk


IC_CAP = 6
IC_MD = 4
IC_PD = 0.5
IC_PARAM = {'pur_cost': -0.25, 'sto_cost': -0.1, 'sale_pri': 1}


# Initialize the different ranges

RISK_PARAM_MIN_BELOW_1 = -0.3
RISK_PARAM_MAX_BELOW_1 = 0.9

RISK_PARAM_MIN_ABOVE_1 = 1.1
RISK_PARAM_MAX_ABOVE_1 = 2.3


IC_HZN = [5, 10, 15, 20, 25, 30]
TRUE_RISK_PARAM = np.array([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2], dtype=np.float)

def param_sampler_below_1():
    return uniform_sampler(RISK_PARAM_MIN_BELOW_1, RISK_PARAM_MAX_BELOW_1)

def prox_below_1(x, t):
    return proj(x, RISK_PARAM_MIN_BELOW_1, RISK_PARAM_MAX_BELOW_1)

def param_sampler_above_1():
    return uniform_sampler(RISK_PARAM_MIN_ABOVE_1, RISK_PARAM_MAX_ABOVE_1)

def prox_above_1(x, t):
    return proj(x, RISK_PARAM_MIN_ABOVE_1, RISK_PARAM_MAX_ABOVE_1)


overall_nll=np.zeros((len(IC_HZN), len(TRUE_RISK_PARAM)))
overall_pred_risk_param = np.zeros_like(overall_nll)
overall_true_risk_param = np.zeros_like(overall_nll)
overall_ic_hzn = np.zeros_like(overall_nll, dtype=np.int)

# Loop through each of the horizons
for i, ic_hzn in reversed(list(enumerate(IC_HZN))):
    for j, true_risk_param in enumerate(TRUE_RISK_PARAM):
        print('Risk Parameter: ' + str(true_risk_param) + ', Horizon: ' + str(ic_hzn))
        overall_true_risk_param[i, j] = true_risk_param
        overall_ic_hzn[i, j] = ic_hzn

        ice = InventoryControlEnvironment(ic_hzn, IC_CAP, IC_MD, IC_PD)
        icr = InventoryControlRewardWithRisk(IC_PARAM['pur_cost'], IC_PARAM['sto_cost'], IC_PARAM['sale_pri'], true_risk_param)
        ica = SoftmaxAgent(ice, icr)
        trajectory = ica.play((0, 0))

        def fun(param):
            reward = InventoryControlRewardWithRisk(IC_PARAM['pur_cost'], IC_PARAM['sto_cost'], 
                                                    IC_PARAM['sale_pri'], param)
            return nll_mdp(ice, trajectory, reward)
        
        # Solve for below 1
        pred_risk_param_below_1, nll_below_1 = fit(fun, param_sampler_below_1, prox=prox_below_1, max_iter=1000, num_starts=3, step_size=1e-2, tol=1e-4)
        print("done below 1")
        print(nll_below_1)
        
        # Solve at 1
        nll_at_1 = fun(1.0)[0]
        print("done at 1")
        print(nll_at_1)

        # Solve for above 1
        pred_risk_param_above_1, nll_above_1 = fit(fun, param_sampler_above_1, prox=prox_above_1, max_iter=1000, num_starts=3, step_size=1e-2, tol=1e-4)
        print("done above 1")
        print(nll_above_1)

        overall_nll[i, j] = np.amin([nll_below_1[0], nll_at_1[0], nll_above_1[0]])
        print(np.argmin([nll_below_1, nll_at_1, nll_above_1]))
        overall_pred_risk_param[i, j] = np.array([pred_risk_param_below_1[0], 1.0, pred_risk_param_above_1[0]])[np.argmin([nll_below_1, nll_at_1, nll_above_1])]


    print("save")
    np.savez(open('Expanded_Domain_Horizon-20200114.npz', 'wb'), 
            ic_hzn=overall_ic_hzn, true_risk_param=overall_true_risk_param, pred_risk_param=overall_pred_risk_param, nll=overall_nll)


# fig, ax = plt.subplots()
# ax.set_xlabel('Horizon')
# ax.set_ylabel('Error (predicted-true)')
# ax.scatter(ic_hzn, pred_risk_param-true_risk_param)

