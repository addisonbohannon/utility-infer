#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Utility inference from choice data in a sequential decision environment
Author: Addison Bohannon
"""

import numpy as np
import matplotlib.pyplot as plt
from solver import fit
from agent import SoftmaxAgent, MyopicSoftmaxAgent
from utility import uniform_sampler, proj, nll_mdp, nll_iid
from inventory_control import InventoryControlEnvironment, InventoryControlRewardWithRisk


IC_HZN = 5
IC_CAP = 6
IC_MD = 4
IC_PD = 0.5
IC_PARAM = {'pur_cost': -0.25, 'sto_cost': -0.1, 'sale_pri': 1}
RISK_PARAM_MIN = 0.1
RISK_PARAM_MAX = 0.9
        

def param_sampler():
    return uniform_sampler(RISK_PARAM_MIN, RISK_PARAM_MAX)


def prox(x, t):
    return proj(x, RISK_PARAM_MIN, RISK_PARAM_MAX)


ice = InventoryControlEnvironment(IC_HZN, IC_CAP, IC_MD, IC_PD)

# Need to create a loop and index appropriately
true_risk_param = np.arange(RISK_PARAM_MIN, RISK_PARAM_MAX+0.05, 0.05)


predicted_vals = []
nll_vals = []
hzn=[]
true_risk=[]
for ns in range(len(true_risk_param)):
	print('Risk Parameter: ' + str(true_risk_param[ns]) + ', Horizon: ' + str(IC_HZN))
	icr = InventoryControlRewardWithRisk(IC_PARAM['pur_cost'], IC_PARAM['sto_cost'], IC_PARAM['sale_pri'], true_risk_param[ns])
	# Prudent Agent
	ica_prudent = SoftmaxAgent(ice, icr)
	trajectory_prudent = ica_prudent.play((0, 0))
	# Myopic Agent
	ica_myopic = MyopicSoftmaxAgent(ice, icr)
	trajectory_myopic = ica_myopic.play((0, 0))
	# Define negative log-likelihood as a function of the risk preference
	def fun(trajectory, prudent=True):
	    def nll(param):
	        reward = InventoryControlRewardWithRisk(IC_PARAM['pur_cost'], IC_PARAM['sto_cost'], IC_PARAM['sale_pri'], param)
	        if prudent:
	            return nll_mdp(ice, trajectory, reward)
	        else:
	            return nll_iid(ice, trajectory, reward)
	    return lambda param : nll(param)
	# Fit myopic and prudent model to myopic and prudent trajectory
	param_pp, nll_pp = fit(fun(trajectory_prudent, prudent=True), param_sampler, prox=prox, num_starts=3, tol=1e-4, max_iter=1000, step_size=1e-2)
	param_pm, nll_pm = fit(fun(trajectory_prudent, prudent=False), param_sampler, prox=prox, num_starts=3, tol=1e-4, max_iter=1000, step_size=1e-2)
	param_mp, nll_mp = fit(fun(trajectory_myopic, prudent=True), param_sampler, prox=prox, num_starts=3, tol=1e-4, max_iter=1000, step_size=1e-2)
	param_mm, nll_mm = fit(fun(trajectory_myopic, prudent=False), param_sampler, prox=prox, num_starts=3, tol=1e-4, max_iter=1000, step_size=1e-2)

	true_risk.append(true_risk_param[ns])
	hzn.append(IC_HZN)
	predicted_vals.append((param_pp[0],param_pm[0],param_mp[0],param_mm[0]))
	nll_vals.append((nll_pp[0],nll_pm[0],nll_mp[0],nll_mm[0]))
	print(predicted_vals)
	np.savez(open('PrudentVsMyopic_horizon_5.npz', 'wb'), hzn=hzn, true_risk=true_risk, pred_risk=predicted_vals, nll=nll_vals)

