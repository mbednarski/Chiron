import sys
import abc
import copy

from monitor import Monitor

assert sys.version_info >= (3, 5)


class Agent(abc.ABC):
    def __init__(self, env, parameters=None):
        self.env = env
        self._parameters = dict()
        self.monitor = Monitor()
        if parameters is None:
            parameters = self.get_default_parameters()
        self.set_parameters(parameters)

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def _construct(self):
        pass

    def _on_parameters_set(self):
        pass

    def get_parameters(self):
        return copy.deepcopy(self._parameters)

    def set_parameters(self, params):
        if params is None: return
        self._parameters.update(params)
        self._on_parameters_set()
        self._construct()
        self.monitor.write_info(self, self.env)

    @abc.abstractmethod
    def get_default_parameters(self):
        pass

    @abc.abstractmethod
    def run(self, max_steps=None, max_episodes=1000):
        pass
