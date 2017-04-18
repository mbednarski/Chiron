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
