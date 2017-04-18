from chiron.agents.qagent import QAgent
import gym


def test_qagent():
    env = gym.make('FrozenLake-v0')
    qagent = QAgent(env)
    qagent.run(max_steps=100, max_episodes=100)
    qagent.close()
