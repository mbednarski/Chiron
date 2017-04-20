import numpy as np

import chiron.agents.util as au
import gym


def test_get_space_state_is_discrete():
    env1 = gym.make('Copy-v0')
    assert au.space_state_is_discrete(env1)
    env1.close()
    env2 = gym.make('CartPole-v0')
    assert not au.space_state_is_discrete(env2)
    env2.close()


def test_get_action_space_is_discrete():
    env1 = gym.make('CartPole-v0')
    assert au.action_space_is_discrete(env1)
    env1.close()
    env2 = gym.make('MountainCarContinuous-v0')
    assert not au.action_space_is_discrete(env2)
    env2.close()


def naive(N, P, T, theta, u, S, sigma):
    for i in range(P):
        for j in range(N):
            T[i, j] = theta[j, i] - u[i]

    for i in range(P):
        for j in range(N):
            S[i, j] = np.divide(np.square(T[i, j].T) - np.square(sigma[i]),
                                sigma[i])

    return T, S


def vectorized(N, P, T, theta, u, S, sigma):
    T = theta.T - np.repeat(np.array([u]).T,N,axis=1)

    S = np.divide(np.square(T.T) - np.square(sigma),
                  sigma).T

    return T, S


def test_xd():
    N = 500
    P = 250
    theta = np.random.normal(size=(N, P))
    T = np.random.normal(size=(P, N))
    S = np.random.normal(size=(P, N))
    sigma = np.random.normal(size=(P,))
    u = np.random.normal(size=(P,))

    nT, nS = naive(N, P, T, theta, u, S, sigma)
    vT, vS = vectorized(N, P, T, theta, u, S, sigma)

    assert np.array_equal(nT, vT)
    assert np.array_equal(nS, vS)
