import sys
import abc
import copy

assert sys.version_info >= (3, 5)


class Agent(abc.ABC):
    def __init__(self, env, parameters):
        self.parameters = dict()
        self.set_parameters(parameters)
        self.env = env

    @property
    @abc.abstractmethod
    def name(self):
        pass

    def get_parameters(self):
        return copy.deepcopy(self.parameters)

    def set_parameters(self, params):
        self.parameters.update(params)

    @abc.abstractmethod
    def run(self, max_steps=None, max_episodes=1000):
        pass