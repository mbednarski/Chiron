from agents.agent import Agent
import agents.util as au

import itertools
import numpy as np
import gym

from featurizers import NullFeaturizer, Featurizer
from policies import EpsilonGreedyPolicy, GreedyPolicy, Policy


class QAgent(Agent):
    def __init__(self, env, parameters=None):
        assert au.space_state_is_discrete(env)
        assert au.action_space_is_discrete(env)
        super().__init__(env, parameters)

    # noinspection PyAttributeOutsideInit
    def _on_parameters_set(self):
        super()._on_parameters_set()
        self.alpha = self._parameters['alpha']
        self.gamma = self._parameters['gamma']
        self.initial_Q = self._parameters['initial_Q']

    # noinspection PyAttributeOutsideInit
    def _construct(self):
        self.featurizer = Featurizer.create_featurizer(self._parameters['featurizer'], self.env)

        self.n_actions = self.env.action_space.n
        self.Q = np.ones((self.n_actions, self.featurizer.shape))

        self.policy = Policy.create_policy(self._parameters['policy'], lambda s: self.Q[:, s])

        self.monitor.add_buffer('episode_reward')

    def get_default_parameters(self):
        return {
            'alpha': 0.5,
            'gamma': 0.8,
            'initial_Q': 0.0,
            'policy': 'greedy',
            'featurizer': 'null'
        }

    def episode(self, i_episode, max_steps):
        episode_reward = 0
        obs = self.env.reset()
        state = self.featurizer.transform(obs)

        for t in itertools.count():
            if t == max_steps:
                print('aborting...')
                return

            action = self.policy.select_action(state)

            new_obs, reward, done, _ = self.env.step(action)

            new_state = self.featurizer.transform(new_obs)

            self.learn(state, new_state, action, reward, done)

            state = new_state
            episode_reward += reward

            if done:
                print('Episode {} finished with score {} after {} steps'.format(i_episode, episode_reward, t))
                self.monitor.append_episode('episode_reward', episode_reward)
                return

    def learn(self, state, new_state, action, reward, done):
        if done:
            delta = reward - self.Q[action, state]
        else:
            delta = reward + self.gamma * np.max(self.Q[:, new_state]) - self.Q[action, state]

        self.Q[action, state] += self.alpha * delta

    def run(self, max_steps=None, max_episodes=1000):
        for i_episode in range(max_episodes):
            self.episode(i_episode, max_steps=max_steps)
        self.monitor.dump()

    @property
    def name(self):
        return 'Q-learning agent'


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    agent = QAgent(env)
    agent.set_parameters({
        'alpha': 0.2,
        'gamma': 0.99,
        'initial_Q': 1.0,
        'policy': 'greedy',
        'featurizer': 'null'
    })
    agent.run(max_steps=10000, max_episodes=1000)
    env.close()
