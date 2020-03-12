import gym
import numpy as np
env = gym.make('CartPole-v0')

pvariance = 0.1 # variance of initial parameters
ppvariance = 0.02 # variance of perturbations
nhiddens = 5 # number of hidden neurons

ninputs = env.observation_space.shape[0]

if (isinstance(env.action_space, gym.spaces.box.Box)):
    noutputs = env.action_space.shape[0]
else:
    noutputs = env.action_space.n
# initialize the training parameters randomly by using a gaussian distribution with average 0.0 and variance 0.1
# biases (thresholds) are initialized to 0.0
W1 = np.random.randn(nhiddens,ninputs) * pvariance # first layer
W2 = np.random.randn(noutputs, nhiddens) * pvariance # second layer
b1 = np.zeros(shape=(nhiddens, 1)) # bias first layer
b2 = np.zeros(shape=(noutputs, 1)) # bias second layer

for episodes in range(20): # Run for number of episodes
    fitness = 0
    observation = env.reset() # Reset enviournment at each step, it reset the cart pole at random position. 
    for _ in range(200): # Number of steps, for which the cart should be stable

        observation.resize(ninputs,1)
        Z1 = np.dot(W1, observation) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = np.tanh(Z2)

        if (isinstance(env.action_space, gym.spaces.box.Box)):
            action = A2
        else:
            action = np.argmax(A2)

        env.render()
        observation, reward, done, info = env.step(action)

        if(reward==1): # Reward is the fitness parameter that is used to evaluate the performance
            fitness = fitness + reward
    env.close()
    print('Reward for episode '+str(episodes)+' = ', fitness)


