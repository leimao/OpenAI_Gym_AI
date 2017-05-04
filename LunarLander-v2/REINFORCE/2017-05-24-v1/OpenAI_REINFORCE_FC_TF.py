'''
REINFORCE Monte Carlo Policy Gradient AI Player
Author: Lei Mao
Date: 5/2/2017
Introduction: 
The REINFORCE_AI used REINFORCE, one of the Monte Carlo Policy Gradient methods, to optimize the AI actions in certain environment. It is extremely complicated to implement the loss function of REINFORCE in Keras. Tensorflow, though it takes time to construct the neural network, makes it easier to customize different loss functions.
'''

import os
import numpy as np
import tensorflow as tf

GAMMA = 0.99 # decay rate of past observations
LEARNING_RATE = 0.005 # learning rate in deep learning
RAND_SEED = 0 # random seed
SAVE_PERIOD = 100 # period of time steps to save the model
LOG_PERIOD = 100 # period of time steps to save the log of training
MODEL_DIR = 'model/' # path for saving the model
LOG_DIR = 'log/' # path for saving the training log

np.random.seed(RAND_SEED)
tf.set_random_seed(RAND_SEED)

class OpenAI_REINFORCE_FC():

    def __init__(self, num_actions, num_features):
    
        # Initialize the number of player actions available in the game
        self.num_actions = num_actions
        # Initialize the number of features in the observation
        self.num_features = num_features
        # Initialize the model
        self.model = self.REINFORCE_FC_Setup()
        # Initialize tensorflow session
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # Initialize the episode number
        self.episode = 0
        # Initialize episode replays used for caching game transitions in one single episode
        self.episode_observations = list() # observation feature list
        self.episode_actions = list() # one-hot encoded action
        self.episode_rewards = list() # immediate reward


    def Softmax_Cross_Entropy(softmax_label, softmax_pred):

        # Calculate cross entropy for softmaxed label and prediction matrices 
        # This function is not used in Tensorflow version of the code
        return (-1.) * np.dot(softmax_label, np.log(softmax_pred.T))

    def One_Hot_Encoding(labels, num_class):

        # Transform labels to one-hot encoded array
        # This function is not used in Tensorflow version of the code
        matrix_encoded = np.zeros(len(labels), num_class, dtype = np.bool)
        matrix_encoded[np.arrange(len(labels)), labels] = 1

        return matrix_encoded
    
    def REINFORCE_FC_Setup(self):

        # Set up REINFORCE Tensorflow environment
        with tf.name_scope('inputs'):

            self.tf_observations = tf.placeholder(tf.float32, [None, self.num_features], name = 'observations')
            self.tf_actions = tf.placeholder(tf.int32, [None,], name = 'num_actions')
            self.tf_values = tf.placeholder(tf.float32, [None,], name = 'state_values')

        # FC1
        fc1 = tf.layers.dense(
            inputs = self.tf_observations,
            units = 16,
            activation = tf.nn.tanh,  # tanh activation
            kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer = tf.constant_initializer(0.1),
            name='FC1'
        )

        # FC2
        fc2 = tf.layers.dense(
            inputs = fc1,
            units = 32,
            activation = tf.nn.tanh,  # tanh activation
            kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer = tf.constant_initializer(0.1),
            name='FC2'
        )

        # FC3
        logits = tf.layers.dense(
            inputs = fc2,
            units = self.num_actions,
            activation = None,
            kernel_initializer = tf.random_normal_initializer(mean=0, stddev = 0.3),
            bias_initializer = tf.constant_initializer(0.1),
            name='FC3'
        )

        # Softmax
        self.action_probs = tf.nn.softmax(logits, name='action_probs')

        with tf.name_scope('loss'):

            # To maximize (log_p * V) is equal to minimize -(log_p * V)
            # Construct loss function mean(-(log_p * V)) to be minimized by tensorflow
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = self.tf_actions) # this equals to -log_p
            self.loss = tf.reduce_mean(neg_log_prob * self.tf_values)

        with tf.name_scope('train'):

            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

    def REINFORCE_FC_Restore(self):

        # Restore the trained model
        self.saver.restore(self.sess, MODEL_DIR + 'AI_model')

    def Store_Transition(self, observation, action, reward):

        # Store game transitions used for updating the weights in the Policy Neural Network
        self.episode_observations.append(observation)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def Clear_Episode_Replays(self):

        # Clear game transitions
        self.episode_observations = list()
        self.episode_actions = list()
        self.episode_rewards = list()

    def Calculate_Value(self):

        # The estimate of v(St) is updated in the direction of the complete return:
        # Gt = Rt+1 + gamma * Rt+2 + gamma^2 * Rt+3 + ... + gamma^(T-t+1)RT;
        # where T is the last time step of the episode.
        state_values = np.zeros_like(self.episode_rewards)
        state_values[-1] = self.episode_rewards[-1]
        for t in reversed(range(0, len(self.episode_rewards)-1)):
            state_values[t] = GAMMA * state_values[t+1] + self.episode_rewards[t]

        # Normalization to help the control of the gradient estimator variance
        state_values -= np.mean(state_values)
        state_values /= np.std(state_values)

        return state_values

    def REINFORCE_FC_Train(self):

        # Train model using data from one episode
        inputs = np.array(self.episode_observations)
        state_values = self.Calculate_Value()

        # Start gradient descent
        _, train_loss = self.sess.run([self.optimizer, self.loss], feed_dict = {
        self.tf_observations: np.vstack(self.episode_observations),
        self.tf_actions: np.array(self.episode_actions), 
        self.tf_values: state_values})
        
        # Print train_loss
        print('Episode train loss: %f' %train_loss)
        
        # Save training log routinely
        if self.episode == 0:
            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)
            # Create training log file
            fhand = open(LOG_DIR + 'training_log.txt', 'w')
            fhand.write('EPISODE\tLoss')
            fhand.write('\n')
            fhand.close()
            # Create training parameters file
            fhand = open(LOG_DIR + 'training_parameters.txt', 'w')
            fhand.write('RAND_SEED\t' + str(RAND_SEED) + '\n')
            fhand.write('NUM_FEATURES\t' + str(self.num_features) + '\n')
            fhand.write('GAMMA\t' + str(GAMMA) + '\n')
            fhand.write('LEARNING_RATE\t' + str(LEARNING_RATE) + '\n')
            fhand.write('SAVE_PERIOD\t' + str(SAVE_PERIOD) + '\n')
            fhand.write('LOG_PERIOD\t' + str(LOG_PERIOD) + '\n')
            fhand.close()

        if self.episode % LOG_PERIOD == 0:
            fhand = open(LOG_DIR + 'training_log.txt', 'a')
            fhand.write(str(self.episode) + '\t' + str(train_loss))
            fhand.write('\n')
            fhand.close()
        
        # Save model routinely
        if self.episode % SAVE_PERIOD == 0:
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)
            # Save all the trained models
            # saver.save(self.sess, MODEL_DIR + 'AI_model', global_step = self.episode)
            # Save the latest trained models
            self.saver.save(self.sess, MODEL_DIR + 'AI_model')

        # Clear episode replays after training for one episode
        self.Clear_Episode_Replays()

        return train_loss

    def AI_Action(self, observation):

        # Calculate action probabilities when given observation
        prob_weights = self.sess.run(self.action_probs, feed_dict = {self.tf_observations: observation[np.newaxis, :]})

        # Randomly choose action according to the probabilities
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())

        return action
