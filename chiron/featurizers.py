import abc


class InvalidFeaturizerError(ValueError):
    """Such featurizer does not exist"""


class Featurizer(abc.ABC):
    @abc.abstractmethod
    def transform(self, observation):
        """Transform an observation into features"""

    @property
    @abc.abstractmethod
    def shape(self):
        """Return features size"""

    @classmethod
    def create_featurizer(cls, name, env):
        if name == 'null':
            return NullFeaturizer(env)

        raise InvalidFeaturizerError(name)


class NullFeaturizer(Featurizer):
    def __init__(self, env):
        self.observation_space_shape = env.observation_space.n
        self.features_shape = self.observation_space_shape

    def transform(self, observation):
        return observation

    @property
    def shape(self):
        return self.features_shape
