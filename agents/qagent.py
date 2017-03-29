from agents.agent import Agent


class QAgent(Agent):
    def run(self, max_steps=None, max_episodes=1000):
        pass

    @property
    def name(self):
        return 'Q-learning agent'

