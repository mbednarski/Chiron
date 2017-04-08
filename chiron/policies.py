import abc

import numpy as np
import scipy
from scipy.misc import logsumexp
import scipy.optimize


class InvalidPolicyError(ValueError):
    """Such policy does not exist"""


class Policy(abc.ABC):
    @abc.abstractmethod
    def select_action(self, state):
        pass

    def get_monitor_data(self):
        return {}

    def on_episode_begin(self, i_episode):
        pass

    def on_episode_end(self, i_episode):
        pass

    @classmethod
    def create_policy(cls, name, *args, **kwargs):
        if name == 'greedy':
            return GreedyPolicy(*args)
        if name == 'epsilon-greedy':
            return EpsilonGreedyPolicy(*args, **kwargs)
        if name == 'boltzmann':
            return BoltzmannPolicy(*args, **kwargs)
        if name == 'mellowmax':
            return MellowmaxPolicy(*args, **kwargs)

        raise InvalidPolicyError(name)


class MellowmaxPolicy(Policy):
    def __init__(self, action_scorer, omega):
        self.action_scorer = action_scorer
        self.omega = omega

    def mellowmax(self, values):
        n = values.shape[0]
        return (logsumexp(self.omega * values) - np.log(n)) / self.omega

    def select_action(self, state):
        values = self.action_scorer(state)
        mm = self.mellowmax(values)

        batch_adv = values - mm
        batch_beta = np.empty_like(batch_adv)

        def f(y, adv):
            return np.sum(np.exp(y * adv) * adv)

        for i in range(values.shape[0]):
            a = batch_adv[i]
            try:
                beta = scipy.optimize.brentq(f, a=-100, b=100, args=(a,))
            except ValueError:
                beta = 0
            batch_beta[i] = beta

        return self.softmax(batch_beta * values)

    def softmax(self, x):
        y = x - np.max(x)
        y = np.exp(y)
        y /= np.sum(y)

        return np.random.choice(range(y.shape[0]), p=y)


class BoltzmannPolicy(Policy):
    def __init__(self, action_scorer, temperature=10.0, decay=0.99, threshold=0.1):
        self.threshold = threshold
        self.action_scorer = action_scorer
        self.decay = decay
        self.temperature = temperature
        self.initial_temperature = temperature

    def select_action(self, state):
        values = self.action_scorer(state)
        if self.temperature <= self.threshold:
            candidates = np.argwhere(values == np.amax(values))
            return np.random.choice(candidates.flatten())

        temp_normalized = values / self.temperature
        maxTNormed = np.max(temp_normalized)

        # summ = np.sum(np.exp(temp_normalized - maxTNormed))
        # lsum = np.log(summ)

        lsum = logsumexp(temp_normalized - maxTNormed)

        probs = np.exp(temp_normalized - maxTNormed - lsum)

        return np.random.choice(range(probs.shape[0]), p=probs)

    def on_episode_begin(self, i_episode):
        self.temperature = self.initial_temperature * np.power(self.decay, i_episode)

    def on_episode_end(self, i_episode):
        pass


class GreedyPolicy(Policy):
    def __init__(self, action_scorer):
        self.action_scorer = action_scorer

    def select_action(self, state):
        values = self.action_scorer(state)
        candidates = np.argwhere(values == np.amax(values))
        return np.random.choice(candidates.flatten())


class EpsilonGreedyPolicy(Policy):
    def __init__(self, action_scorer, epsilon=1.0, epsilon_decay=0.99, threshold=0.0):
        self.threshold = threshold
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.action_scorer = action_scorer
        self.epsilon = epsilon

    def get_monitor_data(self):
        return {'epsilon': self.epsilon}

    def select_action(self, state):
        values = self.action_scorer(state)

        if self.epsilon > self.threshold and np.random.rand() < self.epsilon:
            return np.random.randint(low=0, high=values.shape[0])

        candidates = np.argwhere(values == np.amax(values))
        return np.random.choice(candidates.flatten())

    def on_episode_begin(self, i_episode):
        self.epsilon = self.initial_epsilon * np.power(self.epsilon_decay, i_episode)
