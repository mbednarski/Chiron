from __future__ import print_function, division

import copy
import abc
import itertools

import numpy as np

import gym


class Featurizer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def transform(self, observation):
        """Transform an observation into features"""


class NullFeaturizer(Featurizer):
    def __init__(self, env):
        self.observation_space_shape = env.observation_space.n
        self.features_shape = self.observation_space_shape

    def transform(self, observation):
        return observation


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


class EpsilonGreedyPolicy(Policy):
    def __init__(self, action_scorer):
        self.action_scorer = action_scorer
        self.epsilon = 1.0
        self.decay = 0.99

    def select_action(self, state, eps):
        if np.random.rand() < eps:
            return np.random.randint(low=0, high=4)

        values = self.action_scorer(state)
        candidates = np.argwhere(values == np.amax(values))
        # return np.random.choice(candidates.flatten())
        return np.argmax(values)


class QAgent(object):
    def __init__(self, env):
        self.env = env
        self.params = dict()
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n

        self.alpha = 0.0
        self.gamma = 0.0
        self.initial_Q = 0.0
        self.epsilon = 0.0
        self.max_episodes = 10000
        self.max_steps = 10000
        self.wins = 0

        self.history = np.zeros(self.max_episodes)

        self.set_parameters(self.get_default_parameters())
        self.featurizer = NullFeaturizer(self.env)
        self.Q = np.ones((self.n_actions, self.featurizer.features_shape)) * self.initial_Q


        self.policy = EpsilonGreedyPolicy(lambda s: self.Q[:, s])

    def get_default_parameters(self):
        return {
            'alpha': 0.3,
            'gamma': 0.99,
            'initial_Q': 1.0,
            'epsilon': 1.0
        }

    def set_parameters(self, params):
        self.params.update(params)
        self.alpha = self.params['alpha']
        self.gamma = self.params['gamma']
        self.initial_Q = self.params['initial_Q']
        self.epsilon = self.params['epsilon']

    def get_parameters(self):
        return copy.deepcopy(self.params)

    def episode(self, i_episode):
        self.epsilon *= 0.0
        creward = 0
        obs = self.env.reset()
        state = self.featurizer.transform(obs)

        for t in itertools.count():
            if t == self.max_steps:
                print('aborting...')
                return

            if i_episode == self.max_episodes - 1:
                self.env.render()

            explore = np.random.random() < self.epsilon
            if explore:
                action =  np.random.randint(0, self.n_actions)
            else:
                action = np.argmax(self.Q[:, state])

            new_obs, reward, done, _ = self.env.step(action)

            new_state = self.featurizer.transform(new_obs)

            self.learn(state, new_state, action, reward, done)

            state = new_state
            creward += reward

            if done:
                self.history[i_episode] = creward
                print('Episode {} finished with score {} after {} steps {}'.format(i_episode, reward, t, self.epsilon))
                if reward == 1:
                    self.wins += 1
                return

    def learn(self, state, new_state, action, reward, done):
        if done:
            delta = reward - self.Q[action, state]
        else:
            delta = reward + self.gamma * np.max(self.Q[:, new_state]) - self.Q[action, state]

        self.Q[action, state] += self.alpha * delta

    def game(self):
        for i_episode in range(self.max_episodes):
            self.episode(i_episode)

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    agent = QAgent(env)
    agent.game()
    print(agent.Q)
    U = agent.Q.mean(axis=0).reshape((4, 4))
    for row in range(4):
        print(U[row])
    print(agent.wins)

    q = agent.Q
    for y in range(4):
        for x in range(4):
            a = np.argmax(q[:, y*4 +x])
            print(a, end='')
        print()

    score = np.mean(agent.history[-100:])
    print(score)

