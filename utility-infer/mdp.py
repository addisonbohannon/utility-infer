#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Utility inference from choice data in a sequential decision environment
Author: Addison Bohannon
"""

from abc import ABC, abstractmethod
import numpy as np

RNG = np.random.default_rng()


class MarkovDecisionEnvironment(ABC):
    
    @property
    @abstractmethod
    def horizon(self):
        ...

    @abstractmethod
    def transition_prob(self, next_state, current_state, action, curr_rnd):
        ...

    @abstractmethod
    def is_terminal(self, state, rnd):
        ...

    @abstractmethod
    def state_iterator(self):
        ...

    @abstractmethod
    def enumerate_actions(self, state, rnd):
        ...


class Reward(ABC):
    
    @property
    @abstractmethod
    def param_shape(self):
        ...

    @abstractmethod
    def value(self, state, action):
        ...

    @abstractmethod
    def gradient(self, state, action):
        ...
