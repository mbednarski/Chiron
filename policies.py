from __future__ import print_function, division

import abc

import numpy as np


class Policy(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def select_action(self, state):
        pass

    def episode_end(self):
        pass


class BoltzmannPolicy(Policy):
    def __init__(self, action_scorer, temperature=20, decay=0.99):
        self.action_scorer = action_scorer
        self.decay = decay
        self.temperature = temperature
        self.episodes = 0

    def select_action(self, state):
        if self.episodes > 3000:
            return np.argmax(self.action_scorer(state))
        probs = np.divide(
            np.exp(np.divide(self.action_scorer(state), self.temperature)),
            np.sum(
                np.exp(np.divide(self.action_scorer(state), self.temperature))
            )
        )
        return np.random.choice(range(probs.shape[0]), p=probs)

    def episode_end(self):
        self.episodes += 1
        self.temperature *= self.decay
        if self.temperature < 0.01:
            self.temperature = 0.01


class GreedyPolicy(Policy):
    def __init__(self, action_scorer):
        self.action_scorer = action_scorer

    def select_action(self, state):
        values = self.action_scorer(state)
        candidates = np.argwhere(values == np.amax(values))
        return np.random.choice(candidates.flatten())


class EpsilonGreedyPolicy(Policy):
    def __init__(self, action_scorer, decay=0.99):
        self.decay = decay
        self.action_scorer = action_scorer
        self.epsilon = 1.0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(low=0, high=4)

        values = self.action_scorer(state)
        candidates = np.argwhere(values == np.amax(values))
        return np.random.choice(candidates.flatten())

    def episode_end(self):
        self.epsilon *= self.decay
