from agents.agent import Agent


class QAgent(Agent):
    def get_default_parameters(self):
        return {
            'alpha' : 0.3,
            'gamma' : 0.99,
            'initial_Q': 0.0
        }


    def run(self, max_steps=None, max_episodes=1000):
        pass

    @property
    def name(self):
        return 'Q-learning agent'

