import agents.util as au
import gym


def test_get_space_state_is_discrete():
    env1 = gym.make('Copy-v0')
    assert au.space_state_is_discrete(env1)
    env2 = gym.make('CartPole-v0')
    assert not au.space_state_is_discrete(env2)
