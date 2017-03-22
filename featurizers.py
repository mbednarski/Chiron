from __future__ import print_function, division

import abc


class Featurizer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def transform(self, observation):
        """Transform an observation into features"""

    @property
    @abc.abstractmethod
    def shape(self):
        """Return features size"""


class NullFeaturizer(Featurizer):
    def __init__(self, env):
        self.observation_space_shape = env.observation_space.n
        self.features_shape = self.observation_space_shape

    def transform(self, observation):
        return observation

    @property
    def shape(self):
        return self.features_shape
