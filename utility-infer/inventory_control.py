#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Utility inference from choice data in a sequential decision environment
Author: Addison Bohannon
"""

from itertools import product
import numpy as np
import scipy.stats as ss
from mdp import MarkovDecisionEnvironment, Reward

RNG = np.random.default_rng()


def constant_relative_risk_utility(x, r):
    if x == 0.0:
        return 0.0
    elif r == 1.0:
        return np.sign(x) * np.log(np.abs(x))
    else:
        return np.sign(x) * np.abs(x)**(1 - r) / (1 - r)


def derivative_constant_relative_risk_utility(x, r):
    if x == 0.0:
        return 0.0
    elif r == 1.0:
        return np.sign(x) * RNG.random()
    else:
        return np.sign(x) * np.abs(x)**(1 - r) * (r * np.log(np.abs(x)) + 1 - np.log(np.abs(x))) / (1 - r)**2


def power_utility(x, r):
    if x == 0.0:
        return 0.0
    else:
        return np.sign(x) * np.abs(x)**(1 - r)


def derivative_power_utility(x, r):
    if x == 0.0:
        return 0.0
    else:
        return -np.sign(x) * np.abs(x)**(1 - r) * np.log(np.abs(x))


class InventoryControlEnvironment(MarkovDecisionEnvironment):
    horizon = None

    def __init__(self, horizon, capacity, max_demand, prob_demand):
        self.horizon = horizon
        self.capacity = capacity
        self.max_demand = max_demand
        self.prob_demand = prob_demand

    def transition_prob(self, next_state, curr_state, action, curr_rnd):
        curr_supply, curr_demand = curr_state
        next_supply, next_demand = next_state
        if next_supply != np.maximum(0, curr_supply - curr_demand) + action:
            prob = 0
        else:
            prob = ss.binom.pmf(next_demand, self.max_demand, self.prob_demand)
        return prob

    def is_terminal(self, rnd):
        return rnd == self.horizon - 1

    def state_iterator(self):
        return product(range(self.capacity + 1), range(self.max_demand + 1))

    def enumerate_actions(self, state, rnd):
        supply, demand = state
        available_capacity = self.capacity - np.maximum(0,supply - demand)
        if available_capacity == 0:
            return [0]
        else:
            return np.arange(available_capacity)
    
    
class InventoryControlRewardWithRisk(Reward):
    param_shape = (1,)
    
    def __init__(self, pur_cost, stor_cost, sale_pri, param, utility='crra'):
        self.param = np.array(param, dtype=np.float, ndmin=1)
        if utility == 'crra':
            self.utility = constant_relative_risk_utility
            self.derivative_utility = derivative_constant_relative_risk_utility
        elif utility == 'power':
            self.utility = power_utility
            self.derivative_utility = derivative_power_utility
        else:
            raise ValueError('Not a valid utility function: ' + str(utility))
        self.pur_cost, self.stor_cost, self.sale_pri = pur_cost, stor_cost, sale_pri
        
    def payoff(self, state, action):        
        supply, demand = state
        inventory = np.maximum(0, supply - demand)
        sales = demand + np.minimum(0, supply - demand)
        return self.pur_cost * action + self.stor_cost * inventory + self.sale_pri * sales        
        
    def value(self, state, action):
        return self.utility(self.payoff(state, action), self.param[0])
    
    def gradient(self, state, action):        
        return np.array(self.derivative_utility(self.payoff(state, action), self.param[0]))
