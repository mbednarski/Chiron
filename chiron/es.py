from __future__ import print_function, division

import gym

import numpy as np
import matplotlib as plt
import logging
import matplotlib.pyplot as plt

logging.disable(logging.CRITICAL)
np.seterr('raise')
plt.ion()
problem = 'CartPole-v1'
env = gym.make(problem)

validation_env = gym.make(problem)
validation_env = gym.wrappers.Monitor(validation_env, directory='/tmp/es/2', force=True)

n_features = env.observation_space.shape[0]
n_actions = env.action_space.n

P = (n_features * n_actions) + (n_actions)  # number of parameters
N = 200  # number of histories

T = np.zeros((P, N))
S = np.zeros((P, N))

alpha = 0.00005
alpha_sigma = alpha * 0.1
alpha_u = alpha
HISTORY_SIZE = 50

u = 0.0
sigma = 0.1
b = 0.0

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def unpack(model):
    shapes = [
        (n_features, n_actions),
        (1, n_actions),
        # (hidden_layer_size, output_layer_size),
        # (1, output_layer_size),
    ]
    result = []
    start = 0
    for i, offset in enumerate(np.prod(shape) for shape in shapes):
        result.append(model[start:start + offset].reshape(shapes[i]))
        start += offset
    return result



def choose_action(a1):
    return np.argmax(a1)
    # probs = softmax(a1[0])
    return np.random.choice(np.arange(n_actions), p=probs)


def model(theta, state):
    w, b = unpack(theta)

    z = state.dot(w) + b
    a1 = np.tanh(z)

    return choose_action(a1)


def evaluate_policy(theta, env=None):
    if env is None:
        env = gym.make(problem)
    creward = 0
    state = env.reset()

    while True:
        action = model(theta, state)

        new_state, reward, done, _ = env.step(action)
        state = new_state
        creward += reward

        if done:
            return creward



u = np.repeat(u, P)
assert u.shape == (P,)
sigma = np.repeat(sigma, P)
assert sigma.shape == (P,)

r_history = [0.0]
val_history = [0.0]
mean_history = [0.0]
cross_history = []

m = 0.0

for _ in range(500):
    epsilon = np.zeros((N,P))
    theta_plus = np.zeros((N, P))
    theta_minus = np.zeros((N, P))
    r_plus = np.zeros(N)
    r_minus = np.zeros(N)

    for n in range(N):
        epsilon[n, :] = np.random.normal(0, np.square(sigma))
        theta_plus[n, :] = u + epsilon[n, :]
        theta_minus[n, :] = u - epsilon[n, :]
        r_plus[n] = evaluate_policy(theta_plus[n])
        r_minus[n] = evaluate_policy(theta_minus[n])
        if r_plus[n] > m: m = r_plus[n]
        if r_minus[n] > m: m = r_minus[n]

    for i in range(P):
        for j in range(N):
            T[i, j] = epsilon[j, i]

    for i in range(P):
        for j in range(N):
            S[i, j] = np.divide(
                np.square(epsilon[j,i]) - np.square(sigma[i]),
                sigma[i]
            )

    r_t = np.divide(
        r_plus - r_minus,
        2 * m - r_plus - r_minus
    ).T

    r_s = np.divide(
        r_plus + r_minus - 2*b,
        2*(m-b)
    ).T

    u += alpha_u * np.square(sigma) *np.matmul(T, r_t)
    sigma += alpha_sigma * np.square(sigma) * np.matmul(S, r_s)

    b = np.mean(r_history[-200:])
    r_history.append(np.mean((r_plus + r_minus)/2))

    val_score = evaluate_policy(u, validation_env)
    val_history.append(val_score)

    print(np.mean(val_score))
    plt.clf()
    plt.plot(val_history)
    plt.plot(r_history)
    plt.legend(['validation', 'mean'])
    plt.pause(0.05)

    # b = np.mean(r_history[-HISTORY_SIZE:])
validation_env.close()

