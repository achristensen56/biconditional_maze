import tensorflow as tf
import numpy as np
import os
from time import gmtime, strftime
from tensorflow.python.ops import variables as vars_
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import logging_ops
import random

class DQAgent():
	'''
	Q(s, a) = r + y(max(Q(s', a')))

	'''

	def __init__(self, eps = .3, gam = .99, state_space = 19):

		#exploration factor
		self.epsilon = eps
		#discount factor
		self.y = gam

		self.experience_buffer = ExperienceBuffer(buffer_size = 100)

		#set up log directory for tensorboard logging
		self.log_dir = os.path.join('./logs', strftime("%Y%m%d%H%M%S"))

		print("the log directory is: {}".format(self.log_dir))
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)


		tf.reset_default_graph()


		self.num_hidden = 4
		self.state_space = state_space
		self.num_actions = 4

		self.inputs = tf.placeholder(shape = [None, self.state_space], dtype = tf.float32 )
		
		with tf.name_scope('weights'):
			self.W1 = tf.Variable(tf.random_uniform([self.state_space, self.num_hidden], 0, 0.01))
			self.variable_summaries(self.W1)

		with tf.name_scope('output'):
			self.Qout = tf.matmul(self.inputs, self.W1)
			self.variable_summaries(self.Qout)

			self.predict = tf.argmax(self.Qout, 1)
			self.variable_summaries(self.predict)
			
		with tf.name_scope('loss'):
			self.nextQ = tf.placeholder(shape = [None, self.num_actions], dtype = tf.float32)
			self.variable_summaries(self.nextQ)

			self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
			self.variable_summaries(self.loss)

		with tf.name_scope('train'):
			self.train_step = tf.train.AdamOptimizer(2e-3).minimize(self.loss)
			
			variables = vars_.trainable_variables()
			self.gradients = tf.train.AdamOptimizer(2e-3).compute_gradients(self.loss, variables)

			for gradient, variable in self.gradients:
#			    if isinstance(gradient, ops.IndexedSlices):
#			    	grad_values = gradient.values
#			    else:
#			    	grad_values = gradient

			    self.variable_summaries(gradient)
#			    logging_ops.histogram_summary(variable.name, variable)
#			    logging_ops.histogram_summary(variable.name + "/gradients", grad_values)
#			    logging_ops.histogram_summary(variable.name + "/gradient_norm",
#			    clip_ops.global_norm([grad_values]))

			#self.variable_summaries(self.train_step)

		self.sess = tf.Session()
		self.merged = tf.summary.merge_all()
		self.train_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
		self.sess.run(tf.global_variables_initializer())


	def variable_summaries(self, var):
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)

	def select_action(self, state):

		a, allQ = self.sess.run([self.predict, self.Qout], feed_dict = {	
			self.inputs: state})

		if np.random.rand(1) < self.epsilon:
			a[0] = np.random.randint(self.num_actions)

		return a, allQ

	def update_network(self, r, old_state, new_state, a, allQ, write_summary = False):
		Q_new = self.sess.run(self.Qout, feed_dict = {self.inputs: new_state})
		maxQnew = np.max(Q_new)

		targetQ = allQ

		targetQ[0, a[0]] = r + self.y*maxQnew

		if not write_summary:
			self.sess.run(self.train_step, feed_dict = {self.inputs: old_state, self.nextQ: targetQ})
		else:
			summary, _ = self.sess.run([self.merged, self.train_step], feed_dict = {self.inputs: old_state, self.nextQ: targetQ})
			self.train_writer.add_summary(summary)

	def update_network_batch(self, num_episodes, trace_length):
		episodes = self.experience_buffer.sample(batch_size = num_episodes, trace_length= trace_length)

		#unpack episodes so we can do GD.
		oldstate = np.vstack(episodes[:, 0])
		newstate = np.vstack(episodes[:, 3])
		rewards = np.vstack(episodes[:, 2])
		actions = np.vstack(episodes[:, 1])

		print(oldstate.shape, newstate.shape, rewards.shape, actions.shape)

		Q_old = self.sess.run(self.Qout, feed_dict = {self.inputs:oldstate})
		Q_new = self.sess.run(self.Qout, feed_dict = {self.inputs:newstate})

		print(Q_old.shape, Q_new.shape)

		targetQ = Q_old

		targetQ[range(num_episodes*trace_length), actions] = rewards + self.y*np.max(Q_new, axis = 1)
		self.sess.run(self.train_step, feed_dict = {self.inputs:oldstate, self.nextQ:targetQ})

		print("damn we actually did a batch update!")



#	def write_summary(state, target_Q):
#		summary = self.sess.run(self.merged, feed_dict = {self.inputs: state, self.nextQ: targetQ})

class tabular_agent():
	def __init__(self, state_space = 104, num_actions = 4):
		self.Q = np.zeros([state_space, num_actions])
		self.lr = .85
		self.gamma = .99
		self.eps = .1
	def select_action(self, state):

		state = np.argmax(state)

		
		#use an epsilon greedy policy to select an action
		if np.random.rand(1) < self.eps:
			a = np.random.randint(4)
		else:
			a = np.argmax(self.Q[state, :])


		return [a], self.Q[state, a]


	def update_network(self, r, old_state, new_state, a, oldQ, blah = False):
		old_state = np.argmax(old_state)
		new_state = np.argmax(new_state)


		self.Q[old_state,a] += self.lr*(r + self.gamma*np.max(self.Q[new_state, :]) -  self.Q[old_state, a])

class DQAgent2():
	'''
	Q(s, a) = r + y(max(Q(s', a')))

	multilayer 

	TODO: annealing of epsilon at every N epochs
	TODO: add experience replay 

	'''

	def __init__(self, eps = .3, gam = .99, state_space = 19):

		#exploration factor
		self.epsilon = eps
		#discount factor
		self.y = gam

		#set up log directory for tensorboard logging
		self.log_dir = os.path.join('./logs', strftime("%Y%m%d%H%M%S"))

		print("the log directory is: {}".format(self.log_dir))
		if not os.path.exists(self.log_dir):
			os.makedirs(os.path.join(self.log_dir))


		tf.reset_default_graph()

		self.experience_buffer = ExperienceBuffer(buffer_size = 10)
		self.num_hidden = 20
		self.state_space = state_space
		self.num_actions = 4

		self.inputs = tf.placeholder(shape = [None, self.state_space], dtype = tf.float32 )
		
		with tf.name_scope('layer_1'):

			with tf.name_scope('variables'):
				self.W1 = tf.Variable(tf.random_uniform([self.state_space, self.num_hidden], 0, 0.01))
				self.b1 = tf.Variable(tf.random_uniform([self.num_hidden], 0, 0.01))
				self.variable_summaries(self.W1)
				self.variable_summaries(self.b1)

			with tf.name_scope('hidden1'):
				self.h1 = tf.nn.relu(tf.matmul(self.inputs, self.W1) + self.b1)

		with tf.name_scope('layer_2'):

			with tf.name_scope('variables'):
				self.W2 = tf.Variable(tf.random_uniform([self.num_hidden, self.num_actions], 0, 0.01))
				self.b2 = tf.Variable(tf.random_uniform([self.num_actions]))


		with tf.name_scope('output'):
			self.Qout = tf.matmul(self.h1, self.W2) + self.b2
			self.variable_summaries(self.Qout)

			self.predict = tf.argmax(self.Qout, 1)
			#self.variable_summaries(self.predict)
			
		with tf.name_scope('loss'):
			self.nextQ = tf.placeholder(shape = [None, self.num_actions], dtype = tf.float32)
			self.variable_summaries(self.nextQ)

			self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
			self.variable_summaries(self.loss)

		with tf.name_scope('train'):
			self.train_step = tf.train.AdamOptimizer(2e-3).minimize(self.loss)
			
			variables = vars_.trainable_variables()
			self.gradients = tf.train.AdamOptimizer(2e-3).compute_gradients(self.loss, variables)

			for gradient, variable in self.gradients:
			    self.variable_summaries(gradient)


		self.sess = tf.Session()
		self.merged = tf.summary.merge_all()
		self.train_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()

	def save_variables(self, i, model_name = './models/bigger_deep_Q'):
		self.saver.save(sess = self.sess, save_path = model_name, global_step = i)
	
	def write_summary(self, old_state, targetQ):
		summary = self.sess.run(self.merged, feed_dict = {self.inputs: old_state, self.nextQ: targetQ})
		self.train_writer.add_summary(summary)

	def variable_summaries(self, var):
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)

	def select_action(self, state):

		a, allQ = self.sess.run([self.predict, self.Qout], feed_dict = {	
			self.inputs: state})

		if np.random.rand(1) < self.epsilon:
			a[0] = np.random.randint(self.num_actions)

		return a, allQ

	def update_network(self, r, old_state, new_state, a, oldQ, i):
		Q_new = self.sess.run(self.Qout, feed_dict = {self.inputs: new_state})
		maxQnew = np.max(Q_new)

		targetQ = oldQ

		targetQ[0, a[0]] = r + self.y*maxQnew

		self.sess.run(self.train_step, feed_dict = {self.inputs: old_state, self.nextQ: targetQ})

		if (i % 200 == 0):
			
			self.write_summary(old_state, targetQ)
			self.save_variables(i)

	def update_network_batch(self, num_episodes, trace_length):
		episodes = self.experience_buffer.sample(batch_size = num_episodes, trace_length= trace_length)

		#unpack episodes so we can do GD.
		oldstate = np.vstack(episodes[:, 0])
		newstate = np.vstack(episodes[:, 3])
		rewards = np.vstack(episodes[:, 2])
		actions = np.vstack(episodes[:, 1])

		print(oldstate.shape, newstate.shape, rewards.shape, actions.shape)

		Q_old = self.sess.run(self.Qout, feed_dict = {self.inputs:oldstate})
		Q_new = self.sess.run(self.Qout, feed_dict = {self.inputs:newstate})

		print(Q_old.shape, Q_new.shape)

		targetQ = Q_old

		targetQ[range(num_episodes*trace_length), actions] = rewards + self.y*np.max(Q_new, axis = 1)
		self.sess.run(self.train_step, feed_dict = {self.inputs:oldstate, self.nextQ:targetQ})

		print("damn we actually did a batch update!")

#	def write_summary(state, target_Q):
#		summary = self.sess.run(self.merged, feed_dict = {self.inputs: state, self.nextQ: targetQ})


class DRLAgent():
	'''
	'''
	def __init__(self, eps = .5, gam = .9, state_space = 17):
		'''
		13 spatial positions, 2 context, 2 spatial

		'''

		#exploration factor
		self.epsilon = eps
		#discount factor
		self.y = gam

		#set up log directory for tensorboard logging
		self.log_dir = os.path.join('./logs', strftime("%Y%m%d%H%M%S"))

		print("the log directory is: {}".format(self.log_dir))
		if not os.path.exists(self.log_dir):
			os.makedirs(os.path.join(self.log_dir))


		tf.reset_default_graph()

		self.num_hidden = 20
		self.state_space = state_space
	
		self.num_steps = 10
		self.num_out = self.num_actions  = 4

		with tf.name_scope('inputs'):
			self.inputs = tf.placeholder(shape = [10, 17], dtype = tf.float32)
			self.spatial = self.inputs[:, 0:13]
			self.context = self.inputs[:, 13:]

		with tf.name_scope('spatial_embedding'):
			self.W1 = tf.Variable(tf.random_uniform([13, self.num_hidden], 0, 0.01))
			self.b1 = tf.Variable(tf.random_uniform([self.num_hidden], 0, 0.01))
			self.variable_summaries(self.W1)
			self.spatial_embedding = tf.nn.relu(tf.matmul(self.spatial, self.W1) + self.b1)

		with tf.name_scope('context_embedding'):
			self.W4 = tf.Variable(tf.random_uniform([4, self.num_hidden], 0, 0.01))
			#self.b4 = tf.Variable(tf.random_uniform([self.num_hidden], 0, 0.01))
			self.variable_summaries(self.W4)
			self.context_embedding = tf.matmul(self.context, self.W4)

		self.initial_state_1 = self.state1 = tf.tuple([tf.zeros([1, self.num_hidden]), tf.zeros([1, self.num_hidden])])
		self.initial_state_2 = self.state2 = tf.tuple([tf.zeros([1, self.num_hidden]), tf.zeros([1, self.num_hidden])])

		self.W2 = tf.Variable(tf.random_uniform([self.num_hidden, self.num_hidden], 0, 0.01))
		self.b2 = tf.Variable(tf.random_uniform([self.num_hidden], 0, 0.01))

		self.W3 = tf.Variable(tf.random_uniform([self.num_hidden, self.num_out], 0, 0.01))
		self.b3 = tf.Variable(tf.random_uniform([self.num_out], 0, 0.01))

		with tf.variable_scope('first_layer'):
			self.lstm1 = tf.contrib.rnn.BasicLSTMCell(self.num_hidden)
			self.output1, self.state1 = self.lstm1(tf.reshape(self.spatial_embedding[0, :], [1, self.num_hidden]), self.state1)

		self.input2 = tf.nn.relu(tf.matmul(self.output1, self.W2) + self.b2)
		
		with tf.variable_scope('second_layer'):
			self.lstm2 = tf.contrib.rnn.BasicLSTMCell(self.num_hidden)
			self.output2, self.state2 = self.lstm2(self.input2, self.state2)

		for i in range(1, self.num_steps):

			with tf.variable_scope('first_layer', reuse = True):
				self.output1, self.state1 = self.lstm1(tf.reshape(self.spatial_embedding[i, :], [1, self.num_hidden]), self.state1)
			
			self.input2 = tf.nn.relu(tf.matmul(self.output1, self.W2) + self.context_embedding[i, :] + self.b2)

			with tf.variable_scope('second_layer', reuse = True):
				self.output2, self.state2 = self.lstm2(self.input2, self.state2)


		with tf.name_scope('output'):
			self.Qout = tf.nn.relu(tf.matmul(self.output2, self.W3) + self.b3)
			self.variable_summaries(self.Qout)

			self.predict = tf.argmax(self.Qout, 1)

		with tf.name_scope('loss'):
			self.nextQ = tf.placeholder(shape = [1, self.num_actions], dtype = tf.float32)
			self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
			self.variable_summaries(self.loss)

		with tf.name_scope('train'):
			self.train_step = tf.train.AdamOptimizer(2e-3).minimize(self.loss)

		self.sess = tf.Session()
		self.merged = tf.summary.merge_all()
		self.train_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
		self.sess.run(tf.global_variables_initializer())


	def variable_summaries(self, var):
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)

	def save_variables(self, i, model_name = './models/bigger_deep_Q'):
		self.saver.save(sess = self.sess, save_path = model_name, global_step = i)
	
	def write_summary(self, old_state, targetQ):
		summary = self.sess.run(self.merged, feed_dict = {self.inputs: old_state, self.nextQ: targetQ})
		self.train_writer.add_summary(summary)

	def select_action(self, state):

		#print state.shape
		a, allQ = self.sess.run([self.predict, self.Qout], feed_dict = {self.inputs:state})



		if np.random.rand(1) < self.epsilon:
			a[0]= np.random.randint(self.num_actions)
		return a, allQ

	def update_network(self, r, old_state, new_state, a, allQ, write_summary = False):
		Q_new = self.sess.run(self.Qout, feed_dict = {self.inputs: new_state})
		maxQnew = np.max(Q_new)

		targetQ = allQ

		targetQ[0, a[0]] = r + self.y*maxQnew

		if not write_summary:
			self.sess.run(self.train_step, feed_dict = {self.inputs: old_state, self.nextQ: targetQ})
		else:
			summary, _ = self.sess.run([self.merged, self.train_step], feed_dict = {self.inputs: old_state, self.nextQ: targetQ})
			self.train_writer.add_summary(summary)


class ExperienceBuffer():
	'''
	code for experience buffer borrowed from @awjulian
	'''
	def __init__(self, buffer_size = 1000):
		self.buffer = []
		self.buffer_size = buffer_size

	def add(self, experience):
		print("we added an experience to the buffer, current num episodes: {}".format(len(self.buffer)))
		if len(self.buffer) + 1 >= self.buffer_size:
			self.buffer[0:(1+len(self.buffer)) - self.buffer_size]= []
		self.buffer.append(experience)

	def sample(self, batch_size, trace_length):
		sampled_episodes = random.sample(self.buffer, batch_size)
		sampledTraces = []

		for episode in sampled_episodes:

			point = np.random.randint(len(episode)  - trace_length)
			#we're then permuting samples to remove statistical dependency btwn
			#samples. We'll stop doing this when we are using a DRQN
			if point >= 0:
				sampledTraces.append(np.random.permutation(episode[point:point + trace_length]))
			else:
				batch_size -= 1

		sampledTraces = np.array(sampledTraces)
		print(sampledTraces.shape, batch_size, trace_length)
		return np.reshape(sampledTraces, [batch_size*trace_length, 4])

