#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Utility inference from choice data in a sequential decision environment
Author: Addison Bohannon
"""

from abc import ABC, abstractmethod
from random import choices
import numpy as np
from utility import solve_mdp, solve_mdp_myopic

RNG = np.random.default_rng()


class Agent(ABC):

    def __init__(self, mde, reward):
        self.mde = mde
        self.reward = reward

    def play(self, init_state):
        state, rnd = init_state, 0
        action = self.policy(state, rnd)
        trajectory = [(state, action, rnd)]
        while not self.mde.is_terminal(rnd):
            state, rnd = self.transition(state, action, rnd)
            action = self.policy(state, rnd)
            trajectory.append((state, action, rnd))
        return trajectory
    
    def transition(self, state, action, rnd):
        return choices([state for state in self.mde.state_iterator()], 
                       weights=[self.mde.transition_prob(next_state, state, action, rnd) for next_state in
                                [state for state in self.mde.state_iterator()]])[0], rnd + 1
    
    @abstractmethod
    def policy(self, state, rnd):
        ...


class SoftmaxAgent(Agent):
    
    def __init__(self, mde, reward, t=1):
        Agent.__init__(self, mde, reward)
        self.t = t
        _, _, self.Q, _ = solve_mdp(mde, reward)

    def policy(self, state, rnd):
        available_actions = self.mde.enumerate_actions(state, rnd)
        return choices(available_actions,
                       weights=np.exp([self.t * self.Q[rnd, state, action] for action in available_actions]))[0]
    

class MyopicSoftmaxAgent(SoftmaxAgent):
    
    def __init__(self, mde, reward, t=1):
        Agent.__init__(self, mde, reward)
        self.t = t
        _, _, self.Q, _ = solve_mdp_myopic(mde, reward)
