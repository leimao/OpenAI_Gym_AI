
import gym
from gym import wrappers
import os
import argparse
import numpy as np
from OpenAI_DQN_FC import OpenAI_DQN_FC

ENV_NAME = 'CartPole-v0' # name of the environment
RAND_SEED = 0 # random seed for generating psudorandom numbers
LOG_DIR = 'log/' # path for saving the training log
TEST_DIR = 'test/' # path for saving the test log
RECORD_DIR = 'record/' # path for saving the env record
RECORD_FILENAME = ENV_NAME + '-experiment'
EPISODE_MAX = 2000 # maximum number of episodes
API_KEY = '' # API key for submission

env = gym.make(ENV_NAME)
env.seed(RAND_SEED)

def Train_Model(env = env):
    
    # Initialize AI for training
    num_actions = env.action_space.n
    num_indicators = env.observation_space.shape[0]
    AI_player = OpenAI_DQN_FC(num_actions = num_actions, num_indicators = num_indicators, rand_seed = RAND_SEED, mode = 'train')
    
    # Initialize episode reward log
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    fhand = open(LOG_DIR + 'episode_reward_log.txt', 'w')
    fhand.write('EPISODE\tTIME_STEPS_TAKEN\tTOTAL_REWARD')
    fhand.write('\n')
    fhand.close()
    
    # Start recording env
    env = wrappers.Monitor(env, RECORD_DIR + RECORD_FILENAME, force = True)

    # AI starts training
    episode = 0
    while episode < EPISODE_MAX:
        # Keep training until episode >= EPISODE_max
        observation = env.reset()
        AI_player.Current_State_Initialze(observation = observation)
        t = 0
        episode_reward = 0
        while True:  
            # Keep training until Luna Lander lands or crashes
            env.render() # Turn on visualization
            # print('time step: %d' % AI_player.time_step)
            action = AI_player.AI_Action()
            action_index = np.argmax(action) # binary to numeric
            observation, reward, done, info = env.step(action_index)
            AI_player.Q_FC_Train(action = action, reward = reward, observation = observation, terminal = done)
            t += 1
            episode_reward += reward
            if done:
                # Print episode information
                print("#" * 50)
                print("Episode: %d" % episode)
                print("Episode reward: %f" % episode_reward)
                print("Episode finished after {} timesteps".format(t+1))
                print("Total time step: %d" % AI_player.time_step)
                
                # Save episode information
                fhand = open(LOG_DIR + 'episode_reward_log.txt', 'a')
                fhand.write(str(episode) + '\t' + str(episode_reward) + '\t' + str(t))
                fhand.write('\n')
                fhand.close()
                
                episode += 1
                break

    # Close environment
    env.close()
    
def Test_Model(env = env):

    # Initialize AI for test
    num_actions = env.action_space.n
    num_indicators = env.observation_space.shape[0]
    AI_player = OpenAI_DQN_FC(num_actions = num_actions, num_indicators = num_indicators, rand_seed = RAND_SEED, mode = 'test')
    AI_player.Load_Model()
    
    # Initialize episode reward log
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
    fhand = open(TEST_DIR + 'episode_reward_log.txt', 'w')
    fhand.write('EPISODE\tTIME_STEPS_TAKEN\tTOTAL_REWARD')
    fhand.write('\n')
    fhand.close()

    # AI starts training
    episode = 0
    while True:
        # Keep testing until hitting 'ctrl + c'
        observation = env.reset()
        AI_player.Current_State_Initialze(observation = observation)
        t = 0
        episode_reward = 0
        while True:  
            # Keep training until Luna Lander lands or crashes
            env.render() # Turn on visualization
            # print('time step: %d' % AI_player.time_step)
            action = AI_player.AI_Action()
            action_index = np.argmax(action) # binary to numeric
            observation, reward, done, info = env.step(action_index)
            AI_player.Current_State_Update(observation = observation)
            
            t += 1
            episode_reward += reward
            if done:
                # Print episode information
                print("#" * 50)
                print("Episode: %d" % episode)
                print("Episode reward: %f" % episode_reward)
                print("Episode finished after {} timesteps".format(t+1))
                print("Total time step: %d" % AI_player.time_step)
                
                # Save episode information
                fhand = open(TEST_DIR + 'episode_reward_log.txt', 'a')
                fhand.write(str(episode) + '\t' + str(episode_reward) + '\t' + str(t))
                fhand.write('\n')
                fhand.close()
                
                episode += 1
                break
                
    # Close environment
    env.close()
    
def Upload():

    # Upload training record
    gym.upload(RECORD_DIR + RECORD_FILENAME, api_key = API_KEY)
    
def main():

    parser = argparse.ArgumentParser(description = 'Designate AI mode')
    parser.add_argument('-m','--mode', help = 'train / test / upload', required = True)
    args = vars(parser.parse_args())

    if args['mode'] == 'train':
        Train_Model(env = env)
    elif args['mode'] == 'test':
        Test_Model(env = env)
    elif args['mode'] == 'upload':
        Upload()  
    else:
        print('Please designate AI mode.')

if __name__ == '__main__':

    main()
    
    
    
    
