from gym.spaces import Discrete


def space_state_is_discrete(env):
    return isinstance(env.observation_space, Discrete)


def action_space_is_discrete(env):
    as_ = env.action_space
    return isinstance(as_, Discrete)
