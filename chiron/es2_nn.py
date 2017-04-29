from __future__ import print_function, division

import gym

import numpy as np
import matplotlib as plt
import logging
import matplotlib.pyplot as plt
import itertools
import multiprocessing as mp
from joblib import Parallel, delayed
import time
from chiron.agents.pepg import FeedForwardNeuralNetPolicy

logging.disable(logging.CRITICAL)
np.seterr('raise')
plt.ion()
problem = 'LunarLander-v2'
env = gym.make(problem)

validation_env = gym.make(problem)
validation_env = gym.wrappers.Monitor(validation_env, directory='/tmp/es/2', force=True)

n_features = env.observation_space.shape[0]
n_actions = env.action_space.n

n_hidden = 32
n_hidden2 = 32

P = (n_features + 1) * n_hidden + (n_hidden + 1) * n_hidden2 + (n_hidden2 + 1) * n_actions  # number of parameters
N = 200  # number of histories

T = np.zeros((P, N))
S = np.zeros((P, N))

alpha = 0.0001
alpha_sigma = alpha * 0.1
alpha_u = alpha
HISTORY_SIZE = 50


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def unpack(model):
    shapes = [
        (n_features, n_hidden),
        (1, n_hidden),
        (n_hidden, n_hidden2),
        (1, n_hidden2),
        (n_hidden2, n_actions),
        (1, n_actions),
    ]
    result = []
    start = 0
    for i, offset in enumerate(np.prod(shape) for shape in shapes):
        result.append(model[start:start + offset].reshape(shapes[i]))
        start += offset
    return result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def choose_action(a1):
    return np.argmax(a1)
    probs = softmax(a1[0])
    return np.random.choice(np.arange(n_actions), p=probs)


def model(theta, state):
    w, b, w2, b2, w3, b3 = unpack(theta)

    z = state.dot(w) + b
    a1 = np.tanh(z)

    z2 = a1.dot(w2) + b2
    a2 = np.tanh(z2)

    z3 = a2.dot(w3) + b3
    a3 = np.tanh(z3)

    action = choose_action(a3)
    # if action != 0:
    #     print('x')

    return action


nn = FeedForwardNeuralNetPolicy(n_features, n_actions, hidden_layers=[32, 32])

def evaluate_policy(theta, env=None, render=False, episode_number=0):
    if env is None:
        env = gym.make(problem)
    creward = 0
    state = env.reset()

    frames = []

    for t in itertools.count():
        # action = model(theta, state)
        action = nn.select_action(state)

        new_state, reward, done, _ = env.step(action)

        if render and episode_number != 0:
            arr = env.render(mode='rgb_array')
            frames.append(arr)

        state = new_state
        creward += reward

        if done:
            if render and episode_number != 0:
                fname = 'vid/lunarlander_{}'.format(episode_number)
                arr = np.array(frames)
                np.savez_compressed(fname, arr,)
            return creward


u = 0.0
sigma = 0.5
b = 0.0

u = np.repeat(u, P)
assert u.shape == (P,)
sigma = np.repeat(sigma, P)
assert sigma.shape == (P,)

r_history = []
val_history = []
mean_history = []
cross_history = []
b_200 = []

test_phase = False
test_iterations = 200


def singleiter(theta):
    r = evaluate_policy(theta)
    return r

for _ in range(150000):
    if test_phase:
        val = evaluate_policy(u, validation_env, render=True)
        test_iterations -= 1
        if test_iterations == 0:
            break
        continue

    theta = np.zeros((N, P))
    r = np.zeros(N)
    assert theta.shape == (N, P)
    starttime = time.time()
    # for n in range(N):
    #     theta[n, :] = np.random.normal(u, sigma)
    for n in range(N):
        theta[n, :] = np.random.normal(u, sigma)
        nn.set_params(theta[n])
        r[n] = evaluate_policy(theta[n])


    # results = Parallel(n_jobs=3)(
    #     delayed(singleiter)(theta[n]) for n in range(N)
    # )

    # r = np.array(results)
    # theta = np.array([x[0] for x in results])

    elapsed = time.time() - starttime
    print('Elapsed time: {}'.format(elapsed))

    T = theta.T - np.repeat(np.array([u]).T, N, axis=1)

    S = np.divide(np.square(T.T) - np.square(sigma),
                  sigma).T

    mean_history.append(np.mean(r))

    r = r - b
    r = r.T

    val_score = evaluate_policy(u, validation_env, render=True, episode_number=_ + 1)
    r_history.append(val_score)

    b = np.mean(r_history[-HISTORY_SIZE:])
    b_200 = np.mean(r_history[-20:])

    if b_200 > 200:
        test_phase = True
        continue

    u += alpha_u * np.matmul(T, r)
    sigma += alpha_sigma * np.matmul(S, r)

    val_history.append(val_score)
    cross_history.append(np.mean(val_history[-100:]))
    print(np.mean(val_score))
    plt.clf()
    plt.plot(val_history)
    plt.plot(mean_history)
    plt.plot(cross_history)
    plt.legend(['validation', 'mean', 'benchmark'])
    plt.savefig('xd.png')
    plt.pause(0.05)

    # b = np.mean(r_history[-HISTORY_SIZE:])
validation_env.close()
