from __future__ import print_function, division

import copy
import abc
import itertools

import numpy as np
# np.seterr(all='raise')
import seaborn as sns

import gym
from gym import wrappers

from featurizers import NullFeaturizer
from policies import GreedyPolicy, EpsilonGreedyPolicy, BoltzmannPolicy


class SarsaAgent(object):
    def __init__(self, env, max_episodes=10000, max_steps=10000):
        self.env = env
        self.params = dict()
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n

        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.history = np.zeros(self.max_episodes)

        self.learning = True

        self.set_parameters(self.get_default_parameters())

    # noinspection PyAttributeOutsideInit
    def construct(self):
        self.featurizer = NullFeaturizer(self.env)
        self.Q = np.ones((self.n_actions, self.featurizer.shape)) * self.initial_Q
        # self.policy = GreedyPolicy(lambda s: self.Q[:, s])
        # self.policy = EpsilonGreedyPolicy(lambda s: self.Q[:, s])
        self.policy = BoltzmannPolicy(lambda s: self.Q[:, s])

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

        action = self.policy.select_action(state)

        for t in itertools.count():
            if t == self.max_steps:
                print('aborting...')
                return

            # action = self.policy.select_action(state)

            new_obs, reward, done, _ = self.env.step(action)
            new_state = self.featurizer.transform(new_obs)

            new_action = self.policy.select_action(new_state)

            # if self.learning:
            self.learn(state, new_state, action, new_action, reward, done)

            state = new_state
            action = new_action
            episode_reward += reward

            if done:
                # if episode_reward == 15.0:
                #     self.learning = False
                self.history[i_episode] = episode_reward
                # print('Episode {} finished with score {} after {} steps'.format(i_episode, reward, t))
                return

    def learn(self, state, new_state, action, new_action, reward, done):
        if done:
            delta = reward - self.Q[action, state]
        else:
            delta = reward + self.gamma * self.Q[new_action, new_state] - self.Q[action, state]

        self.Q[action, state] += self.alpha * delta

    def game(self):
        for i_episode in range(self.max_episodes):
            self.episode(i_episode)
            self.policy.episode_end()
            # self.alpha *= 0.993


if __name__ == '__main__':
    env = gym.make('FrozenLake8x8-v0')
    env = wrappers.Monitor(env, '/tmp/fl/2', force=True)
    agent = SarsaAgent(env, max_episodes=5000)
    agent.set_parameters({
            'alpha': 0.2,
            'gamma': 0.99,
            'initial_Q': 0.0
    })
    agent.game()
    env.close()

    means = np.zeros(agent.max_episodes)
    means2 = np.zeros(agent.max_episodes)
    for e in range(agent.max_episodes):
        means[e] = np.mean(agent.history[e - 100:e])
        means2[e] = np.mean(agent.history[e - 500:e])

    print(np.mean(agent.history))

    print(np.mean(agent.history[-100:]))

    print('max mean')
    print(np.nanmax(means))

    print('max score')
    print(np.nanmax(agent.history))

    # sns.tsplot(agent.history)
    sns.tsplot(means, color='r')
    sns.tsplot(means2, color='g')
    sns.plt.show()


