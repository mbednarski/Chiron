from __future__ import print_function, division

import abc

import numpy as np


class Policy(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def select_action(self, state):
        pass


class GreedyPolicy(Policy):
    def __init__(self, action_scorer):
        self.action_scorer = action_scorer

    def select_action(self, state):
        values = self.action_scorer(state)
        candidates = np.argwhere(values == np.amax(values))
        return np.random.choice(candidates.flatten())
