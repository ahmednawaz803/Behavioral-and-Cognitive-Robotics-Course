import gym
import numpy as np
env = gym.make('CartPole-v0')

pop_size = 20

pvariance = 0.1 
ppvariance = 0.02 
nhiddens = 5 

ninputs = env.observation_space.shape[0]
if (isinstance(env.action_space, gym.spaces.box.Box)):
 noutputs = env.action_space.shape[0]
else:
 noutputs = env.action_space.n

W1 = np.random.randn(nhiddens,ninputs) * pvariance # first layer
W2 = np.random.randn(noutputs, nhiddens) * pvariance # second layer
b1 = np.zeros(shape=(nhiddens,1))
b2 = np.zeros(shape=(noutputs,1))

theta_0 = np.concatenate((W1.flatten(), W2.flatten(), b1.flatten(),b2.flatten())) # A vector to estimate the number of parameters of neural net.

parameter_matrix = np.random.randn(pop_size, len(theta_0)) * pvariance # A matrix of randoms values, each row represents different set of neural net parameters.

r = 0
r_ls = []

variation = lambda x : x + ppvariance

for trials in range(1,5): # The convergence depends upon the randomness of the parameters, so this loop randomly initialize the matrix, if it does not converge.
    for gen in range(1,30):
        r=0
        r_ls = []
        for i_episode in range(pop_size):
            observation = env.reset()
            for _ in range(200):
                observation.resize(ninputs,1)
                Z1 = np.dot(parameter_matrix[i_episode][0:20].reshape((5,4)), observation) + parameter_matrix[i_episode][30:35].reshape((5,1))
                A1 = np.tanh(Z1)
                Z2 = np.dot(parameter_matrix[i_episode][20:30].reshape((2,5)), A1) + parameter_matrix[i_episode][35:37].reshape((2,1))
                A2 = np.tanh(Z2)
                if (isinstance(env.action_space, gym.spaces.box.Box)):
                    action = A2
                else:
                    action = np.argmax(A2)
                env.render()
                observation, reward, done, info = env.step(action) # take a random action
                if(reward==1):
                    r+=1
            env.close()
            print('reward for '+str(i_episode)+'th episode '+str(gen)+'th generation and '+str(trials)+' trial = ', r)
            r_ls.append(r)
            r=0
        index = np.argsort(r_ls)
        print(index)
        for rep in range(0, pop_size//2): # Loop to shuffle the rows depending upon the fitness
            parameter_matrix[index[rep]]=variation(parameter_matrix[index[rep+(pop_size//2)]])
    
        if(gen>10 and sum(r_ls)<80*20): # A condition to observe that the performance is not under-fitted.
            print('The training is skipped.......at gen 11')
            break
        
        if(sum(r_ls)==200*20 ): # Condition to terminate the training, because all the episodes are giving maximum reward!
            print('The training completed at trial '+str(trials))
            break


    if(sum(r_ls)==200*20):
        print('The training completed at trial '+str(trials)+' generation '+str(gen)+'.')
        break

    
    parameter_matrix = np.random.randn(pop_size, len(theta_0)) * pvariance

np.savetxt('parameter_matrix.csv', parameter_matrix, delimiter=',')
np.save('parameter.npy',parameter_matrix)

