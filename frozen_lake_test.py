import gym 
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import DRLAgents as drl

#the point of this file is as a sorta unit test for the DQ Agent

env = gym.make('FrozenLake-v0')

agent = drl.DQAgent(state_space = 16, gam = .99, eps = .1)

num_episode = 2000

jList = []
rList = []
aList = []
for i in range(num_episode):

	s = env.reset()
	rAll = 0
	d = False
	j = 0

	while j < 99:
		j+=1

		#env.render()

		a, allQ = agent.select_action(np.identity(16)[s:s+1])

		aList.append(a[0])

		s1, r, d, _ = env.step(a[0])

		#print(r)

		agent.update_network(r, np.identity(16)[s:s+1], np.identity(16)[s1:s1+1], a, allQ)

		rAll +=r
		s = s1

		if d == True:
			e = 1./((i/50) + 10)
			break
	jList.append(j)
	rList.append(rAll)

print ("Percent of succesful episodes: " + str(sum(rList)/num_episode) + "%")

plt.figure()

plt.subplot(131)
plt.plot(rList)

plt.subplot(132)
plt.plot(jList)

plt.subplot(133)
plt.hist(aList)

plt.show()