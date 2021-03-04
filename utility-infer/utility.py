#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Utility inference from choice data in a sequential decision environment
Author: Addison Bohannon
"""

import numpy as np
from mdp import MarkovDecisionEnvironment, Reward


RNG = np.random.default_rng()


def softmax(x, t=1):
    z = np.exp(t * x)
    return z / np.sum(z)


def uniform_sampler(min_value, max_value, shape=(1,)):
    return min_value + (max_value - min_value) * RNG.random(shape, dtype=np.float)


def proj(x, min_value, max_value):
    x[x <= min_value] = min_value
    x[x >= max_value] = max_value
    return x


def solve_mdp(mde, reward):
    v = dict()
    dv = dict()
    q = dict()
    dq = dict()
    for state in mde.state_iterator():
        for action in mde.enumerate_actions(state, mde.horizon - 1):
            q[mde.horizon - 1, state, action], dq[mde.horizon - 1, state, action] \
                = reward.value(state, action), reward.gradient(state, action)
    for rnd in range(mde.horizon - 1, 0, -1):
        for state in mde.state_iterator():
            available_actions = mde.enumerate_actions(state, rnd)
            best_action = available_actions[np.argmax([q[rnd, state, action] for action in available_actions])]
            v[rnd, state], dv[rnd, state] = q[rnd, state, best_action], dq[rnd, state, best_action]
        for state in mde.state_iterator():
            for action in mde.enumerate_actions(state, rnd - 1):
                q[rnd - 1, state, action] = reward.value(state, action) + sum(
                    [mde.transition_prob(next_state, state, action, rnd - 1) * v[rnd, next_state] for next_state in
                     mde.state_iterator()])
                dq[rnd - 1, state, action] = reward.gradient(state, action) + sum(
                    [mde.transition_prob(next_state, state, action, rnd - 1) * dv[rnd, next_state] for next_state in
                     mde.state_iterator()])
    for state in mde.state_iterator():
        available_actions = mde.enumerate_actions(state, 0)
        best_action = available_actions[np.argmax([q[0, state, action] for action in available_actions])]
        v[0, state], dv[0, state] = q[0, state, best_action], dq[0, state, best_action]
    return v, dv, q, dq


def solve_mdp_myopic(mde, reward):
    v = dict()
    dv = dict()
    q = dict()
    dq = dict()
    for rnd in range(mde.horizon):
        for state in mde.state_iterator():
            available_actions = mde.enumerate_actions(state, rnd)
            for action in available_actions:
                q[rnd, state, action] = reward.value(state, action)
                dq[rnd, state, action] = reward.gradient(state, action)
            best_action = available_actions[np.argmax([q[rnd, state, action] for action in available_actions])]
            v[rnd, state], dv[rnd, state] = q[rnd, state, best_action], dq[rnd, state, best_action]
    return v, dv, q, dq
    

def nll_mdp(mde, trajectory, reward):
    assert isinstance(mde, MarkovDecisionEnvironment)
    assert isinstance(reward, Reward)
    _, _, Q, dQ = solve_mdp(mde, reward)
    fun, grad = 0, np.zeros(reward.param_shape)
    for (state, action, rnd) in trajectory:
        available_actions = mde.enumerate_actions(state, rnd)
        temp = [Q[rnd, state, a] for a in available_actions]
        pQ = softmax(temp)
        fun -= np.log(pQ[available_actions == action])
        grad += np.average(np.array([dQ[rnd, state, a] for a in mde.enumerate_actions(state, rnd)]), axis=0,
                           weights=pQ) - dQ[rnd, state, action]
    return fun, grad
    

def nll_iid(mde, trajectory, reward):
    assert isinstance(mde, MarkovDecisionEnvironment)
    assert isinstance(reward, Reward)
    fun, grad = 0, np.zeros(reward.param_shape)
    for (state, action, rnd) in trajectory:
        available_actions = mde.enumerate_actions(state, rnd)
        temp = [reward.value(state, a) for a in available_actions]
        pQ = softmax(temp)
        fun -= np.log(pQ[available_actions == action])
        grad += np.average(np.array([reward.gradient(state, a) for a in mde.enumerate_actions(state, rnd)]), axis=0,
                           weights=pQ) - reward.gradient(state, action)
    return fun, grad