import gym
import itertools as it

env = gym.make('Copy-v0')

max_epochs = 100

for i_epoch in range(max_epochs):
    obs = env.reset()

    creward = 0
    for t in it.count():
        # env.render()
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        creward += reward

        if done:
            print('Epoch {} finished in {} steps, with cumulative reward {}'.format(i_epoch, t, creward))
            break

