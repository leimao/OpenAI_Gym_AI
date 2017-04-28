'''
Deep Q-Learning AI Numerical Game Player
Author: Lei Mao
Date: 4/25/2017
Introduction: 
The DQLN_AI used Deep Q-Learning to study optimal solutions to play a certain game, assuming that the game is a Markov Decision Process. For the training step, the game API exports the game state as numerical string, reward of the game state, and the signal of game termination to the DQLN_AI for learning. For the test step, the DQLN_AI only takes the game state as input and output operations that the DQLN_AI thinks optimal. 
'''

import tensorflow as tf
import numpy as np
import keras
import random
import os
import keras
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model

# Hyperparameters

GAME_STATE_FRAMES = 1  # number of game state frames used as input
GAMMA = 0.9 # decay rate of past observations
EPSILON_INITIALIZED = 0.5 # probability epsilon used to determine random actions
EPSILON_FINAL = 0.01 # final epsilon after decay
BATCH_SIZE = 32 # number of sample size in one minibatch
LEARNING_RATE = 0.0001 # learning rate in deep learning
FRAME_PER_ACTION = 1 # number of frames per action
REPLAYS_SIZE = 1000 # maximum number of replays in cache
TRAINING_DELAY = 1000 # time steps before starting training for the purpose of collecting sufficient replays to initialize training
EXPLORATION_TIME = 10000 # time steps used for decaying epsilon during training before epsilon decreases to zero
SAVING_PERIOD = 5000 # period of time steps to save the model
LOG_PERIOD = 500 # period of time steps to save the log of training
MODEL_DIR = 'model/' # path for saving the model
LOG_DIR = 'log/' # path for saving the training log

class OpenAI_DQN_FC():

    def __init__(self, num_actions, num_indicators, rand_seed, mode):
    
        # Initialize the number of player actions available in the game
        self.num_actions = num_actions
        # Initialize the number of indicators in the state
        self.num_indicators = num_indicators
        # Determine the shape of input to the model
        self.input_shape = self.num_indicators * GAME_STATE_FRAMES
        # Initialize the model
        self.model = self.Q_FC_Setup()
        # Initialize game_replays used for caching game replays
        self.game_replays = deque()
        # Initialize time_step to count the time steps during training
        self.time_step = 0
        # Initialize the mode of AI
        self.mode = mode
        # Initialize epsilon which controls the probability of choosing random actions
        self.epsilon = 0
        if self.mode == 'train':
            self.epsilon = EPSILON_INITIALIZED
        elif self.mode == 'test':
            self.epsilon = 0
        else:
            raise('AI mode error.')
        # Initialize random seed
        self.rand_seed = rand_seed
        # Set seed for psudorandom numbers
        random.seed(self.rand_seed)
        np.random.seed(self.rand_seed)
        
    def Current_State_Initialze(self, observation):
    
        # Initialize current_state with the observation from the game
        self.current_state = np.stack(tuple(observation.tolist() * GAME_STATE_FRAMES), axis = 0)

    def Current_State_Update(self, observation):
    
        # Update current_state with the observation from the game
        self.current_state = np.append(self.current_state[self.num_indicators:], observation.tolist(), axis = 0)
        self.time_step += 1
        
    def State_Format(self, data):
        
        # Add the fourth dimension to the data for Tensorflow
        # For example a single data with dimension of [32,] to [1,32]
        return data.reshape(1, self.input_shape)
        
    def Q_FC_Setup(self):
    
        # Prepare Convolutional Neural Network for Q-Learning
        # Note that we did not use regularization here
        model = Sequential()
        # FC layer_1
        model.add(Dense(20, activation = 'relu', input_dim = self.input_shape))
        # FC layer_2
        #model.add(Dense(64, activation = 'relu'))
        # FC layer_3
        #model.add(Dense(128, activation = 'relu'))
        # FC layer_4
        model.add(Dense(self.num_actions))
        # Optimizer
        optimizer = keras.optimizers.Adam(lr = LEARNING_RATE)
        # Compile the model
        model.compile(loss = keras.losses.mean_squared_error, optimizer = optimizer)
        
        return model
        
    def Q_FC_Train_Batch(self, minibatch):
    
        # Elements in minibatch consists tuple (current_state, state_action, state_reward, next_state, terminal)        
        # Generate inputs and targets from minibatch data and model
        
        # Initialize inputs
        inputs = np.zeros((len(minibatch), self.input_shape))
        targets = np.zeros((len(minibatch), self.num_actions))
        
        # Prepare inputs and calculate targets
        for i in xrange(len(minibatch)):
        
            current_state = minibatch[i][0]
            state_action = minibatch[i][1]
            state_reward = minibatch[i][2]
            next_state = minibatch[i][3]
            terminal = minibatch[i][4]
            
            Qs_current_state = self.model.predict(self.State_Format(current_state))[0]
            Qs_next_state = self.model.predict(self.State_Format(next_state))[0]
            
            inputs[i] = current_state
            targets[i] = Qs_current_state
            
            if terminal:
                targets[i,np.argmax(state_action)] = state_reward
            else:
                targets[i,np.argmax(state_action)] = state_reward + GAMMA * np.max(Qs_next_state)
        
        # Train on batch
        loss = self.model.train_on_batch(inputs, targets)
        
        print('loss: %f' %loss)
        
        # Return training details for print
        return loss, Qs_current_state.astype(np.float), targets[-1].astype(np.float)
        
    def AI_Action(self):
    
        # AI calculate optimal actions for the current state
        state_action = np.zeros(self.num_actions)
        
        if self.time_step % FRAME_PER_ACTION == 0:           
            if random.random() < self.epsilon:
                # Choose random action
                action_index = random.randint(0,self.num_actions-1)
                state_action[action_index] = 1
                              
            else:
                # Choose the optimal action from the model
                Qs = self.model.predict(self.State_Format(self.current_state))[0]
                action_index = np.argmax(Qs)
                state_action[action_index] = 1
        else:
            action_index = 0
            state_action[action_index] = 1
            
        # Update epsilon
        if (self.mode == 'train') and (self.epsilon > 0):
            if (self.time_step >= TRAINING_DELAY) and (self.time_step < (TRAINING_DELAY + EXPLORATION_TIME)):
                self.Epsilon_Update()
            
        return state_action
    
    def Epsilon_Update(self):
        
        # Update epsilon during training
	    self.epsilon -= (EPSILON_INITIALIZED - EPSILON_FINAL)/EXPLORATION_TIME
        
    def Q_FC_Train(self, action, reward, observation, terminal):
    
        # Next state after taking action at current state
        next_state = np.append(self.current_state[self.num_indicators:], observation.tolist(), axis = 0)
        
        # Add the replay to game_replays
        self.game_replays.append((self.current_state, action, reward, next_state, terminal))
        
        # Check game_replays exceeds the size specified
        if len(self.game_replays) > REPLAYS_SIZE:
            # Remove the oldest replay
            self.game_replays.popleft()
        
        # Start training after training delay
        loss = 'NA'
        Qs_predicted_example = 'NA'
        Qs_target_example = 'NA'
        if self.time_step > TRAINING_DELAY:
            # Train Q_CNN on batch
            minibatch = random.sample(self.game_replays, BATCH_SIZE)
            loss, Qs_predicted_example, Qs_target_example = self.Q_FC_Train_Batch(minibatch = minibatch)
            
        # Save model routinely
        if self.time_step % SAVING_PERIOD == 0:
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)
            self.model.save(MODEL_DIR + 'AI_model.h5')

        # Print Training Information
        if self.time_step < TRAINING_DELAY:
            stage = 'DELAY'
        elif (self.time_step >= TRAINING_DELAY) and (self.time_step < (TRAINING_DELAY + EXPLORATION_TIME)):
            stage = 'EXPLORATION'
        else:
            stage = 'TRAINING'
            
        #print('TIME_STEP', self.time_step, '/ STAGE', stage, '/ EPSILON', self.epsilon, '/ ACTION', np.argmax(action), '/ REWARD', reward, '/ Qs_PREDICTED_EXAMPLE', Qs_predicted_example, '/ Qs_TARGET_EXAMPLE', Qs_target_example, '/ Loss', loss)
        
        # Save training log routinely
        if self.time_step == 0:
            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)
            # Create training log file
            fhand = open(LOG_DIR + 'training_log.txt', 'w')
            fhand.write('TIME_STEP\tSTAGE\tEPSILON\tACTION\tREWARD\tQs_PREDICTED_EXAMPLE\tQs_TARGET_EXAMPLE\tLoss')
            fhand.write('\n')
            fhand.close()
            # Create training parameters file
            fhand = open(LOG_DIR + 'training_parameters.txt', 'w')
            fhand.write('RAND_SEED\t' + str(self.rand_seed) + '\n')
            fhand.write('NUM_INDICATORS\t' + str(self.num_indicators) + '\n')
            fhand.write('GAME_STATE_FRAMES\t' + str(GAME_STATE_FRAMES) + '\n')
            fhand.write('GAMMA\t' + str(GAMMA) + '\n')
            fhand.write('EPSILON_INITIALIZED\t' + str(EPSILON_INITIALIZED) + '\n')
            fhand.write('EPSILON_FINAL\t' + str(EPSILON_FINAL) + '\n')
            fhand.write('BATCH_SIZE\t' + str(BATCH_SIZE) + '\n')
            fhand.write('LEARNING_RATE\t' + str(LEARNING_RATE) + '\n')
            fhand.write('FRAME_PER_ACTION\t' + str(FRAME_PER_ACTION) + '\n')
            fhand.write('REPLAYS_SIZE\t' + str(REPLAYS_SIZE) + '\n')
            fhand.write('TRAINING_DELAY\t' + str(TRAINING_DELAY) + '\n')
            fhand.write('EXPLORATION_TIME\t' + str(EXPLORATION_TIME) + '\n')
            fhand.write('SAVING_PERIOD\t' + str(SAVING_PERIOD) + '\n')
            fhand.write('LOG_PERIOD\t' + str(LOG_PERIOD) + '\n')
            fhand.close()

        if self.time_step % LOG_PERIOD == 0:
            fhand = open(LOG_DIR + 'training_log.txt', 'a')
            fhand.write(str(self.time_step) + '\t' + str(stage) + '\t' + str(self.epsilon) + '\t' + str(np.argmax(action)) + '\t' + str(reward) + '\t' + str(Qs_predicted_example) + '\t' + str(Qs_target_example) + '\t' + str(loss))
            fhand.write('\n')
            fhand.close()

        # Update current state
        self.current_state = next_state
        
        # Increase time step
        self.time_step += 1
        
    def Load_Model(self):
    
        # Load the saved model
        self.model = load_model(MODEL_DIR + 'AI_model.h5')
        




            
            
        
            
            
            



