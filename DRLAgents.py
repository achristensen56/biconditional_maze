import tensorflow as tf
import numpy as np

class DQAgent():
	'''
	Q(s, a) = r + y(max(Q(s', a')))
	'''


	def __init__(self, eps = .5, gam = .99, state_space = 19):

		#exploration factor
		self.epsilon = eps

		#discount factor
		self.y = gam

		tf.reset_default_graph()
		#4 for state of platforms
		#2 for current context
		#12 for agent position
		#should I have a different 

		self.num_hidden = 4
		self.state_space = state_space
		self.num_actions = 4

		self.inputs = tf.placeholder(shape = [1, self.state_space], dtype = tf.float32 )
		self.W1 = tf.Variable(tf.random_uniform([self.state_space, self.num_hidden], 0, 0.01))
		self.b1 = tf.Variable(tf.random_uniform([self.num_hidden], 0, 0.01))
		#self.h1 = tf.nn.relu(tf.matmul(self.inputs, self.W1) + self.b1)

		#self.W2 = tf.Variable(tf.random_uniform([self.num_hidden, self.num_actions]))
		#self.b2 = tf.Variable(tf.random_uniform([self.num_actions]))

		#self.Qout = tf.nn.relu(tf.matmul(self.h1, self.W2)  + self.b2)
		
		self.Qout = tf.matmul(self.inputs, self.W1)

		self.predict = tf.argmax(self.Qout, 1)

		self.nextQ = tf.placeholder(shape = [1, self.num_actions], dtype = tf.float32)
		self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))

		self.train_step = tf.train.GradientDescentOptimizer(2e-1).minimize(self.loss)

		self.sess = tf.Session()

		self.sess.run(tf.global_variables_initializer())

	def select_action(self, state):

		a, allQ = self.sess.run([self.predict, self.Qout], feed_dict = {	
			self.inputs: state})

		if np.random.rand(1) < self.epsilon:
			a[0] = np.random.randint(self.num_actions)

		return a, allQ

	def update_network(self, r, old_state, new_state, a, allQ):
		Q_new = self.sess.run(self.Qout, feed_dict = {self.inputs: new_state})
		maxQnew = np.max(Q_new)

		targetQ = allQ

		targetQ[0, a[0]] = r + self.y*maxQnew

		self.sess.run(self.train_step, feed_dict = {self.inputs: old_state, self.nextQ: targetQ})

	def function():
		pass
