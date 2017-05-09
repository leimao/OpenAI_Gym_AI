'''
Sarsa AI Player
Author: Lei Mao
Date: 5/8/2017
Introduction: 
The Sarsa_AI used Sarsa method to optimize the AI actions in certain environment. The critic updates network for the values of the state-action pairs (Q-network).
'''

import os
import numpy as np
import tensorflow as tf

GAMMA = 0.99 # decay rate of past observations
LEARNING_RATE = 0.02 # learning rate
EPSILON = 0.001 # epsilon of e-greedy action sampling
RAND_SEED = 0 # random seed
SAVE_PERIOD = 1000 # period of episode to save the model
LOG_PERIOD = 100 # period of episode to save the log of training
MODEL_DIR = 'model/' # path for saving the model
LOG_DIR = 'log/' # path for saving the training log

np.random.seed(RAND_SEED)
tf.set_random_seed(RAND_SEED)

class OpenAI_Sarsa_FC():

    def __init__(self, num_actions, num_features):
    
        # Initialize the number of player actions available in the game
        self.num_actions = num_actions
        # Initialize the number of features in the observation
        self.num_features = num_features
        # Initialize the model
        self.Q_FC_Setup()
        # Initialize tensorflow session
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # Initialize the episode number
        self.episode = 0
        # Initialize the number of time steps
        self.time_step = 0
        # Initialize the total number of time steps
        self.time_step_total = 0
        # Initialize the training log
        self.Save_Hyperparameters()

    def Save_Hyperparameters(self):
    
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)   
                 
        # Log the hyperparameters used in the training
        fhand = open(LOG_DIR + 'training_parameters.txt', 'w')
        fhand.write('RAND_SEED\t' + str(RAND_SEED) + '\n')
        fhand.write('NUM_FEATURES\t' + str(self.num_features) + '\n')
        fhand.write('NUM_ACTIONS\t' + str(self.num_actions) + '\n')
        fhand.write('GAMMA\t' + str(GAMMA) + '\n')
        fhand.write('LEARNING_RATE\t' + str(LEARNING_RATE) + '\n')
        fhand.write('SAVE_PERIOD\t' + str(SAVE_PERIOD) + '\n')
        fhand.write('LOG_PERIOD\t' + str(LOG_PERIOD) + '\n')
        fhand.close()
               
        # Create training log file
        fhand = open(LOG_DIR + 'training_log.txt', 'w')
        fhand.write('EPISODE\tTIME_STEP_TOTAL\tTRAIN_LOSS')
        fhand.write('\n')
        fhand.close()
        
    def Save_Train_Log(self, train_loss):
    
        # Save training log
        fhand = open(LOG_DIR + 'training_log.txt', 'a')
        fhand.write(str(self.episode) + '\t' + str(self.time_step_total)+ '\t' + str(train_loss))
        fhand.write('\n')
        fhand.close()

    def Q_FC_Setup(self):

        # Set up State value Tensorflow environment
        with tf.variable_scope('critic'):

            self.tf_observation = tf.placeholder(tf.float32, [1, self.num_features], name = 'observation')
            self.tf_reward = tf.placeholder(tf.float32, None, name = 'reward')
            self.tf_Q_value_next = tf.placeholder(tf.float32, None, name = 'Q_value_next')
            self.tf_action = tf.placeholder(tf.int32, None, name = 'action')       

            # FC1
            fc1 = tf.layers.dense(
                inputs = self.tf_observation,
                units = 32,
                activation = tf.nn.relu,
                kernel_initializer = tf.random_normal_initializer(mean = 0, stddev = 0.1),
                bias_initializer = tf.constant_initializer(0.1),
                name='FC1')
                
            # FC2
            logits = tf.layers.dense(
                inputs = fc1,
                units = self.num_actions,
                activation = None,
                kernel_initializer = tf.random_normal_initializer(mean = 0, stddev = 0.1),
                bias_initializer = tf.constant_initializer(0.1),
                name='FC2')
                
            self.Q_values = logits[0]
            self.Q_value = logits[0, self.tf_action]
        
            # Calculate time-dependent error
            self.td_error = self.tf_reward + GAMMA * self.tf_Q_value_next - self.Q_value
            
            # Construct loss funtion
            self.loss = tf.square(self.td_error)

            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)
            
    def Get_Q_Value(self, observation, action):
    
        observation = observation[np.newaxis, :]
        Q_value = self.sess.run([self.Q_value], feed_dict = {self.tf_observation: observation, self.tf_action: action})
                
        return Q_value
                    
    def Q_FC_Train(self, observation, action, reward, done, observation_next, action_next):
    
        observation = observation[np.newaxis, :]
        observation_next = observation_next[np.newaxis, :]
        
        # Calculate the value of the next observation
        if done:
            Q_value_next = 0
        else:
            Q_value_next = self.sess.run(self.Q_value, feed_dict = {self.tf_observation: observation_next, self.tf_action: action_next})

        # Start gradient decent
        td_error, _, train_loss = self.sess.run([self.td_error, self.optimizer, self.loss], feed_dict = {self.tf_observation: observation, self.tf_action: action, self.tf_reward: reward, self.tf_Q_value_next: Q_value_next})

        if done:
            # Save training log routinely
            if self.episode % LOG_PERIOD == 0:
                self.Save_Train_Log(train_loss = train_loss)
            # Save trained model routinely
            if self.episode % SAVE_PERIOD == 0:
                self.Q_FC_Save()            
            
            # Update episode information
            self.Episode_Update()
        
        self.time_step += 1
        self.time_step_total += 1

        return train_loss

    def Q_FC_Save(self, model_name = 'AI_Sarsa'):
    
        # Save the latest trained models
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        self.saver.save(self.sess, MODEL_DIR + model_name)
        print("AI_Sarsa Saved.")
    
    def Q_FC_Restore(self, model_name = 'AI_Sarsa'):

        # Restore the trained FC policy network
        self.saver.restore(self.sess, MODEL_DIR + model_name)
        print("AI_Sarsa Loaded.")

    def Episode_Update(self):

        # Update episode number when the new epsidoe starts
        self.episode += 1
        # Reset time_step to 0 when the new episode starts
        self.time_step = 0
                
    def Get_Action(self, observation):
    
        if np.random.rand() < EPSILON:
            # Choose random action
            action = np.random.randint(0,self.num_actions)
        else:
            # Choose the optimal action from the model
            Q_values = self.sess.run(self.Q_values, feed_dict = {self.tf_observation: observation[np.newaxis, :]})
            action = np.argmax(Q_values)

        return action


















class OpenAI_QACPG_AI():

    def __init__(self, num_actions, num_features):
    
        # Initialize the number of player actions available in the game
        self.num_actions = num_actions
        # Initialize the number of features in the observation
        self.num_features = num_features                
        # Initialize the actor
        self.actor = Actor(num_actions = num_actions, num_features = num_features)
        # Initialize the critic
        self.critic = Critic(num_actions = num_actions, num_features = num_features)                
        # Initialize the episode number
        self.episode = 0
        # Initialize the number of time steps
        self.time_step = 0
        # Initialize the total number of time steps
        self.time_step_total = 0
        # Initialize the training log
        self.Save_Hyperparameters()

    def Episode_Update(self):

        # Update episode number when the new epsidoe starts
        self.episode += 1
        # Reset time_step to 0 when the new episode starts
        self.time_step = 0

    def Save_Hyperparameters(self):
        
        # Log the hyperparameters used in the training
        fhand = open(LOG_DIR + 'training_parameters.txt', 'w')
        fhand.write('RAND_SEED\t' + str(RAND_SEED) + '\n')
        fhand.write('NUM_FEATURES\t' + str(self.num_features) + '\n')
        fhand.write('NUM_ACTIONS\t' + str(self.num_actions) + '\n')
        fhand.write('GAMMA\t' + str(GAMMA) + '\n')
        fhand.write('LEARNING_RATE_ACTOR\t' + str(LEARNING_RATE_ACTOR) + '\n')
        fhand.write('LEARNING_RATE_CRITIC\t' + str(LEARNING_RATE_CRITIC) + '\n')
        fhand.write('SAVE_PERIOD\t' + str(SAVE_PERIOD) + '\n')
        fhand.write('LOG_PERIOD\t' + str(LOG_PERIOD) + '\n')
        fhand.close()
               
        # Create training log file
        fhand = open(LOG_DIR + 'training_log.txt', 'w')
        fhand.write('EPISODE\tTIME_STEP_TOTAL\tTRAIN_LOSS_ACTOR\tTRAIN_LOSS_CRITIC')
        fhand.write('\n')
        fhand.close()
        
    def Save_Train_Log(self, train_loss_actor, train_loss_critic):
    
        # Save training log
        fhand = open(LOG_DIR + 'training_log.txt', 'a')
        fhand.write(str(self.episode) + '\t' + str(self.time_step_total)+ '\t' + str(train_loss_actor) + '\t' + str(train_loss_critic))
        fhand.write('\n')
        fhand.close()
            
    def Load(self):

        # Load trained actor and critic for test
        self.actor.Policy_FC_Restore()
        self.critic.Value_FC_Restore()

    def Train(self, observation, action, reward, done, observation_next, action_next):
    
        # Train the actor policy network
        Q_value = self.critic.Get_Q_Value(observation = observation, action = action)
        train_loss_actor = self.actor.Policy_FC_Train(observation = observation, action = action, Q_value = Q_value)
        # Train the critic value network
        td_error, train_loss_critic = self.critic.Value_FC_Train(observation = observation, action = action, reward = reward, done = done, observation_next = observation_next, action_next = action_next)

        if done:
            # Save training log routinely
            if self.episode % LOG_PERIOD == 0:
                self.Save_Train_Log(train_loss_actor = train_loss_actor, train_loss_critic = train_loss_critic)
            # Save trained model routinely
            if self.episode % SAVE_PERIOD == 0:
                self.actor.Policy_FC_Save()
                self.critic.Value_FC_Save()            
            
            # Update episode information
            self.Episode_Update()
            self.critic.Episode_Update()
            self.actor.Episode_Update()
            
            return train_loss_actor, train_loss_critic
            
        self.time_step += 1
        self.time_step_total += 1
        
        return train_loss_actor, train_loss_critic
    
    def Get_Action(self, observation):

        # Get action instruction from the actor
        return self.actor.Get_Action(observation = observation)

