import numpy as np
import gym
import itertools


class QAgent(object):
    def __init__(self, env):
        self.env = env
        self.parameters = self.get_default_parameters()
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.Q = np.ones((self.n_actions, self.n_states))

    def get_action(self, state):
        explore = np.random.random() < self.parameters['epsilon']
        if explore:
            return np.random.randint(0, self.n_actions)
        return np.argmax(self.Q[:, state])

    def get_maxQ(self, state):
        return np.max(self.Q[:, state])

    def learn(self, state, new_state, action, reward, done):

        if done:
            diff = self.parameters['alpha'] * (reward - self.Q[action, state])
        else:
            diff = self.parameters['alpha'] * (reward + self.parameters['gamma'] * self.get_maxQ(new_state) - self.Q[action, state])

        self.Q[action, state] += diff

    def episode(self):
        state = self.env.reset()
        creward = 0
        for t in itertools.count():
            # self.env.render()
            action = self.get_action(state)

            new_state, reward, done, _ = env.step(action)
            creward += reward
            self.learn(state, new_state, action, reward, done)

            state = new_state

            if done:
                print('Finished after {} steps with score {} eps{}'.format(t, creward, self.parameters['epsilon']))
                # print(self.parameters)
                break

    def run(self):
        for i_episode in range(10000):
            self.new_episode()
            self.episode()

    def get_default_parameters(self):
        return {'alpha': 0.5,
                'gamma': 0.8,
                'epsilon': 0.0}

    def new_episode(self):
        self.parameters['epsilon'] *= 0.99


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    agent = QAgent(env)
    agent.run()
    print(agent.Q)
