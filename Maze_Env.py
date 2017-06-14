import numpy as np
import random
import copy


class Maze():
	'''defines the maze environment, which processes an agents actions, does state transitions, and emits rewards'''
	
	def __init__(self, initial_context = np.random.randint(2), initial_platform_cond = np.random.choice([0, 1, 2, 3]), 
				initial_position = 12, batch_length = 5):
		'''
		environment state definition:

		-----             -----
	   |  0  |			 |  1  |
	   	-----------------------
	   |  4  |  5  |  6  |  7  |
	    -----------------------
	                    |  12  |
	    -----------------------
	   |  8  | 9  |  10  |  11  |
	    -----------------------
	   |  2  |           |  3  |
	    -----             -----


	    state masking for POMDP

	    platform_mask = { 0: [1, 0, 0, 0], 
	    				  1: [0, 1, 0, 0],
	    				  2: [0, 0, 1, 0], 
	    				  3: [0, 0, 0, 1], 
	    				  4: [1, 0, 0, 0], 
	    				  5: [1, 0, 0, 0], 
	    				  6: [0, 1, 0, 0], 
	    				  7: [0, 1, 0, 0], 
	    				  8: [0, 0, 1, 0], 
	    				  9: [0, 0, 1, 0], 
	    				  10: [0, 0, 0, 1], 
	    				  11: [0, 0, 0, 1], 
	    				  12: [0, 0, 0, 0]}



		pretraining: a lick is rewarded

		mouse starts in state 12
		
		bedded context --> gridded platform correct
		cardboard context --> smooth platform correct

		first trial could go to either side, correct platform depending on context
		if correct then switch side, correct platform depending on context
		platforms randomized on opposite side

		if incorrect, no reward, nothing switches just need to find correct platform
		standard task now

		got it right for first time, positions of opposite side randomized. 
		if completely correct, start to line 38
		doesn't count as error if they visit other platform but don't lick

		if goes incorrect platform, and licks
		no reward
		previously rewarded platform goes to 90 degrees
		have to get to correct platform, get reward.   

		length blocks: 30, can drop down to blocks of 10. Some block of 40. 

		'''
		self.batch_correct = 0

		self.error = False

		self.trial_history = []
		
		#context: {0, 1}, 0: bedding, 1:(other context)
		self.current_context = initial_context

		self.platform_initialization = [[0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0]]

		#platform_cond: {0101, 0110, 1001, 1010}, where 0 corresponds to close platform smooth, 1 corresponds to close platform grided
		#condition of platform 0, 1, 2, 3 respectively
		self.platform_cond = copy.deepcopy(self.platform_initialization[initial_platform_cond])

		self.platform_mask = { 0: [1, 0, 0, 0], 
		    				   1: [0, 1, 0, 0],
		    				   2: [0, 0, 1, 0], 
		    				   3: [0, 0, 0, 1], 
		    				   4: [1, 0, 0, 0], 
		    				   5: [1, 0, 0, 0], 
		    				   6: [0, 1, 0, 0], 
		    				   7: [0, 1, 0, 0], 
		    				   8: [0, 0, 1, 0], 
		    				   9: [0, 0, 1, 0], 
		    				  10: [0, 0, 0, 1], 
		    				  11: [0, 0, 0, 1], 
		    				  12: [0, 0, 0, 0]}

		self.current_position = initial_position

		self.vis_platform_cond = (np.array(self.platform_cond) + 1)*np.array(self.platform_mask[self.current_position])

		self.current_target_platform = None

		#e.g. self.platform_mapping[self.context] = goal_platform_type
		self.platform_mapping = [1, 0]

							#up, down, left, right, lick
		self.transition_table = {0: [0, 4, 0, 0], 
								 1: [1, 7, 1, 1],
								 2: [8, 2, 2, 2], 
								 3: [11, 3, 3, 3], 
								 4: [0, 4, 4, 5], 
								 5: [5, 5, 4, 6], 
								 6:	[6, 6, 5, 7], 
								 7:	[1, 12, 6, 7], 
								 8: [8, 2, 8, 9], 
								 9: [9, 9, 8, 10], 
								 10: [10, 10, 9, 11], 
								 11: [12, 3, 10, 11], 
								 12: [7, 11, 12, 12]}

		self.batch_length = batch_length
		self.num_batches = 0
		self.total_trial_num = 0
		self.batch_trial_num = 0
		self.last_rewarded_platform = 0
		self.total_rewards_received = 0
		self.batch_rewards_received = 0
		self.reward_history = []
		self.num_batch_steps = 0


	def reset(self):
		self.__init__(initial_context= np.random.randint(2), initial_platform_cond = np.random.randint(4))
		return (self.current_context, self.vis_platform_cond, self.current_position)

	def initialize_new_batch(self, i):
		
		np.random.seed(i)
		self.reward_history.append(float(self.batch_correct)/ (self.batch_length))
		self.num_batch_steps = 0
		self.batch_rewards_received = 0
		self.batch_trial_num = 0
		self.batch_correct = 0
		self.current_position = 12
		#switch the context using modulo arithmetic
		self.current_context = (self.current_context + 1) % 2
		self.current_target_platform = None
		self.num_batches += 1

		rand = random.SystemRandom().randint(0, 3)

		self.platform_cond = copy.deepcopy(self.platform_initialization[rand])

		self.vis_platform_cond = (np.array(self.platform_cond) + 1)*np.array(self.platform_mask[self.current_position])


		return (self.current_context, self.vis_platform_cond, self.current_position)

	def initialize_new_trial(self, next_pos, i):

		if not self.error:
			self.batch_correct += 1

		self.error = False

		np.random.seed(i)

		self.last_rewarded_platform = next_pos
		self.total_rewards_received += 1
		self.batch_rewards_received += 1

		#initiate next trial
		self.total_trial_num += 1
		self.batch_trial_num += 1
		#return state

		self.current_position = next_pos

		rand = random.SystemRandom().randint(0, 3)

		new_platform_cond = copy.deepcopy(self.platform_initialization[rand])


		if next_pos in [0, 1]:
			#print('should switch bottom')
			self.platform_cond[2:4] = copy.deepcopy(new_platform_cond[2:4])
			self.current_target_platform = self.platform_cond[2:4].index(self.platform_mapping[self.current_context]) + 2
		else:
			#print('should switch top')
			self.platform_cond[0:2] = copy.deepcopy(new_platform_cond[0:2])
			self.current_target_platform = self.platform_cond[0:2].index(self.platform_mapping[self.current_context])

		#print(rand, new_platform_cond, self.platform_cond, self.platform_initialization)

		self.vis_platform_cond = (np.array(self.platform_cond) + 1)*np.array(self.platform_mask[self.current_position])
		return self.current_context, self.vis_platform_cond, self.current_position



	def step(self, action, i):
		'''action 0: up, 1: down, 2: left, 3: right, 4: lick
		   This function implements the main logic for the game. 
		   Currently doesn't require a lick to receive a reward, 
		   although obviously that should be implemented soon. 
		'''

		self.num_batch_steps += 1
		state = None
		d = 0

		next_pos = self.transition_table[self.current_position][action]

		#we check to see that the agent doesn't try to go back to the platform that was just rewarded
		if next_pos == self.last_rewarded_platform:
			next_pos == self.current_position

		if self.batch_trial_num == 0:
			if next_pos in [0, 1, 2, 3]:
				if self.platform_cond[next_pos] == self.platform_mapping[self.current_context]:
					reward = 1
					state = self.initialize_new_trial(next_pos, i)
					
				else:
					reward = 0
					self.error = True
					self.current_position = next_pos
					state = (self.current_context, self.vis_platform_cond, self.current_position)
			else:
				reward = 0
				self.current_position = next_pos
				state = (self.current_context, self.vis_platform_cond, self.current_position)


		else:

			if next_pos == self.current_target_platform:
				reward = 1

				#this logic (initilizing new trial before checking if itwas the last trial in the 
				#block is slightly less efficient but leads to much prettier coe)
				state = self.initialize_new_trial(next_pos, i)


				if self.batch_trial_num > self.batch_length:
					#do start new batch!
					# self.initialize_new_batch
					state = self.initialize_new_batch(i)
					d = 1

			else:
				if next_pos in [0, 1, 2, 3]:
					self.error = True

				reward = 0

				self.current_position = next_pos

				state = (self.current_context, self.vis_platform_cond, self.current_position)


		return state, reward, d

	def render(self, mode = 'human', close = False):
		'''render the current frame of the maze'''
		pass

	def logging():
		''' TODO log the all the actions taken'''
		pass

