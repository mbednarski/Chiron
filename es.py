from __future__ import print_function, division

import gym
import numpy as np
import itertools
from sklearn.neural_network import MLPClassifier

np.random.seed(17)

params = 17*4
alpha = 0.09
sigma = 0.1
theta = [np.random.normal(scale=sigma, size=(17*4,))]
n = 1000

import logging
logging.disable(logging.CRITICAL)


def policy(theta, state, env):
    n_features = env.observation_space.n
    n_actions = env.action_space.n
    n_hidden = n_actions

    x = np.zeros(n_features + 1)
    x[n_features] = 1.0
    x[state] = 1.0

    w = np.zeros((n_features + 1, n_hidden))
    w = theta.reshape((17,4))
    z1 = np.dot(x, w)
    a1 = z1 * (z1 > 0)

    action = np.argmax(a1)
    return action


def F(theta):
    env = gym.make('FrozenLake-v0')
    state = env.reset()
    score = 0
    for step in range(1000):
        action = policy(theta, state, env)
        new_state, reward, done, _ = env.step(action)
        score += reward
        state = new_state
        if done:
            # print('Done with score {}'.format(score))
            return score
    return 0.0


for t in itertools.count():
    epsilons = np.random.normal(size=(n,params))
    F_i = np.zeros(n)
    for i, e in enumerate(epsilons):
        F_i[i] = F(theta[t] + sigma * e)
    sums = np.dot(F_i, epsilons)
    diff = alpha * (1.0/(n * sigma)) * sums
    theta.append(theta[t] + diff)
    print('=====EPISODE END====')
    print('Mean reward: ', np.mean(F_i))


