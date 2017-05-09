import gym
from gym import wrappers
import os
import argparse
import numpy as np
from OpenAI_AC_FC_TF import OpenAI_ACPG_AI

ENV_NAME = 'CartPole-v0' # name of the environment
RENDER = False # show game
RAND_SEED = 0 # random seed for generating psudorandom numbers
LOG_DIR = 'log/' # path for saving the training log
TEST_DIR = 'test/' # path for saving the test log
RECORD_DIR = 'record/' # path for saving the env record
RECORD_FILENAME = ENV_NAME + '-experiment'
EPISODE_MAX = 5000 # maximum number of episodes
API_KEY = '' # API key for submission

env = gym.make(ENV_NAME)
env.seed(RAND_SEED)

def Train_Model(env = env):
    
    # Initialize AI for training
    num_actions = env.action_space.n
    num_features = env.observation_space.shape[0]
    AI_player = OpenAI_ACPG_AI(num_actions = num_actions, num_features = num_features)
    
    # Initialize episode reward log
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    fhand = open(LOG_DIR + 'episode_reward_log.txt', 'w')
    fhand.write('EPISODE\tTIME_STEP_TOTAL\tREWARD_TOTAL')
    fhand.write('\n')
    fhand.close()
    
    # Start recording env
    env = wrappers.Monitor(env, RECORD_DIR + RECORD_FILENAME, force = True)

    # AI starts training
    episode = 0
    while episode < EPISODE_MAX:
        # Keep training until episode >= EPISODE_max
        observation = env.reset()
        t = 0
        episode_reward = 0

        while True:  
            # Keep training until termination
            if RENDER:
                env.render() # Turn on visualization

            action = AI_player.Get_Action(observation = observation)
            observation_next, reward, done, info = env.step(action)
            episode_reward += reward
            
            train_loss_actor, train_loss_critic = AI_player.Train(observation = observation, action = action, reward = reward, done = done, observation_next = observation_next)

            observation = observation_next
            t += 1

            if done:
                # Print episode information
                print("#" * 50)
                print("Episode: %d" % episode)
                print("Episode reward: %f" % episode_reward)
                print("Episode finished after {} timesteps".format(t+1))
                print("Current Actor Train Loss: %f" % train_loss_actor)
                
                # Save episode information
                fhand = open(LOG_DIR + 'episode_reward_log.txt', 'a')
                fhand.write(str(episode) + '\t' + str(t) + '\t' + str(episode_reward))
                fhand.write('\n')
                fhand.close()
                
                episode += 1

                break

    # Close environment
    env.close()

def Test_Model(env = env):

    # Initialize AI for test
    num_actions = env.action_space.n
    num_features = env.observation_space.shape[0]
    AI_player = OpenAI_ACPG_AI(num_actions = num_actions, num_features = num_features)
    AI_player.Load()
    
    # Initialize episode reward log
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
    fhand = open(TEST_DIR + 'episode_reward_log.txt', 'w')
    fhand.write('EPISODE\tTIME_STEP_TOTAL\tREWARD_TOTAL')
    fhand.write('\n')
    fhand.close()

    # AI starts training
    episode = 0

    while True:
        # Keep testing until hitting 'ctrl + c'
        observation = env.reset()
        t = 0
        episode_reward = 0

        while True:
            # Keep training until termination
            env.render() # Turn on visualization

            action = AI_player.Get_Action(observation = observation)
            observation, reward, done, info = env.step(action)

            t += 1
            episode_reward += reward

            if done:
                # Print episode information
                print("#" * 50)
                print("Episode: %d" % episode)
                print("Episode reward: %f" % episode_reward)
                print("Episode finished after {} timesteps".format(t+1))
                
                # Save episode information
                fhand = open(TEST_DIR + 'episode_reward_log.txt', 'a')
                fhand.write(str(episode) + '\t' + str(t) + '\t' + str(episode_reward))
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
