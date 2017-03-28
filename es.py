from __future__ import print_function, division

import gym
import numpy as np
import itertools
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt

np.random.seed(17)

n_inputs = 64
n_outputs = 4

params = (n_inputs+1)*n_outputs
alpha = 0.1
sigma = 0.1
init_range = np.sqrt(6.0 / ((n_inputs+1)+ n_outputs))
theta = [np.random.uniform(-init_range, init_range, size=(params,))]
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
    w = theta.reshape((n_inputs+1,n_outputs))
    z1 = np.dot(x, w)
    # a1 = z1 * (z1 > 0)
    a1 = z1
    action = np.argmax(a1)
    return action


def F(theta):
    env = gym.make('FrozenLake8x8-v0')
    state = env.reset()
    score = 0
    for step in range(1000):
        action = policy(theta, state, env)
        new_state, reward, done, _ = env.step(action)
        reward -= 0.001
        score += reward
        state = new_state
        if done:
            # print('Done with score {}'.format(score))
            return score
    return 0.0

validations = []
means = []
top_ten = []

plt.ion()

for t in itertools.count():
    epsilons = np.random.normal(size=(n,params))
    F_i = np.zeros(n)
    for i, e in enumerate(epsilons):
        F_i[i] = F(theta[t] + sigma * e)

    last_theta = theta[-1]
    best_ten_idx = np.argpartition(F_i, -50)[-50:]
    best_ten_epsilons = epsilons[best_ten_idx]
    best_ten = last_theta + best_ten_epsilons * sigma
    best_ten_avg_policy = np.mean(best_ten, axis=0)
    top_ten += [F(best_ten_avg_policy)]


    sums = np.dot(F_i, epsilons)
    diff = alpha * (1.0/(n * sigma)) * sums
    theta.append(theta[t] + diff)


    print('=====EPISODE END====')
    validation = F(theta[-1])
    print('Mean reward: {:.4f}'.format(np.mean(F_i)))
    print('Validation reward: {:.4f}'.format(validation))
    print('1st percentile reward: {:.4f}'.format(top_ten[-1]))
    validations += [validation]
    means += [np.mean(F_i)]
    plt.clf()
    x = np.linspace(0, t, t+1)
    plt.plot(x, validations, x, means, x, top_ten)
    plt.legend(['val', 'means', '10'])
    plt.pause(0.05)




