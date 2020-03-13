import gym
env = gym.make('CartPole-v0')
print('action space = ', env.action_space)
print('observation space = ', env.observation_space)
print('observation space high = ', env.observation_space.high)
print('observation space low = ', env.observation_space.low)

env = gym.make('Pendulum-v0')
print('action space = ', env.action_space)
print('observation space = ', env.observation_space)
print('observation space high = ', env.observation_space.high)
print('observation space low = ', env.observation_space.low)

env = gym.make('MountainCar-v0')
print('action space = ', env.action_space)
print('observation space = ', env.observation_space)
print('observation space high = ', env.observation_space.high)
print('observation space low = ', env.observation_space.low)