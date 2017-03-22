from __future__ import print_function, division

import copy
import abc
import itertools

import numpy as np
import seaborn as sns

import gym
from gym import wrappers

from featurizers import NullFeaturizer
from policies import GreedyPolicy


class QAgent(object):
    def __init__(self, env, max_episodes=1000, max_steps=10000):
        self.env = env
        self.params = dict()
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n

        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.history = np.zeros(self.max_episodes)

        self.set_parameters(self.get_default_parameters())

    # noinspection PyAttributeOutsideInit
    def construct(self):
        self.featurizer = NullFeaturizer(self.env)
        self.Q = np.ones((self.n_actions, self.featurizer.shape)) * self.initial_Q
        self.policy = GreedyPolicy(lambda s: self.Q[:, s])

    def get_default_parameters(self):
        return {
            'alpha': 0.3,
            'gamma': 0.99,
            'initial_Q': 0.0
        }

    # noinspection PyAttributeOutsideInit
    def set_parameters(self, params):
        self.params.update(params)
        self.alpha = self.params['alpha']
        self.gamma = self.params['gamma']
        self.initial_Q = self.params['initial_Q']

        self.construct()

    def get_parameters(self):
        return copy.deepcopy(self.params)

    def episode(self, i_episode):
        episode_reward = 0
        obs = self.env.reset()
        state = self.featurizer.transform(obs)

        for t in itertools.count():
            if t == self.max_steps:
                print('aborting...')
                return

            action = self.policy.select_action(state)

            new_obs, reward, done, _ = self.env.step(action)

            new_state = self.featurizer.transform(new_obs)

            self.learn(state, new_state, action, reward, done)

            state = new_state
            episode_reward += reward

            if done:
                self.history[i_episode] = episode_reward
                # print('Episode {} finished with score {} after {} steps'.format(i_episode, reward, t))
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
    env = wrappers.Monitor(env, '/tmp/fl/2', force=True)
    agent = QAgent(env, max_episodes=1000)
    agent.set_parameters({
            'alpha': 0.9,
            'gamma': 0.999,
            'initial_Q': 1.0
    })
    agent.game()
    env.close()

    means = np.zeros(agent.max_episodes)
    for e in range(agent.max_episodes):
        means[e] = np.mean(agent.history[e - 100:e])

    print(np.mean(agent.history))

    sns.tsplot(agent.history)
    sns.tsplot(means, color='r')
    sns.plt.show()
