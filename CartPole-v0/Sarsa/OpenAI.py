import gym
from gym import wrappers
import os
import argparse
import numpy as np
from OpenAI_Sarsa_FC_TF import OpenAI_Sarsa_FC

ENV_NAME = 'CartPole-v0' # name of the environment
RENDER = False # show game
RAND_SEED = 0 # random seed for generating psudorandom numbers
LOG_DIR = 'log/' # path for saving the training log
TEST_DIR = 'test/' # path for saving the test log
RECORD_DIR = 'record/' # path for saving the env record
RECORD_FILENAME = ENV_NAME + '-experiment'
EPISODE_MAX = 10000 # maximum number of episodes
EVALUATION_INTERVAL = 100 # evaluation interval to calculate learning performance average
API_KEY = '' # API key for submission

env = gym.make(ENV_NAME)
env.seed(RAND_SEED)

def Train_Model(env = env):
    
    # Initialize AI for training
    num_actions = env.action_space.n
    num_features = env.observation_space.shape[0]
    AI_player = OpenAI_Sarsa_FC(num_actions = num_actions, num_features = num_features)
    
    # Episode Reward Record
    episode_rewards = list()
    episode_reward_average_max = -np.inf
    
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

        action = AI_player.Get_Action(observation = observation)
        while True:  
            # Keep training until termination
            if RENDER:
                env.render() # Turn on visualization

            observation_next, reward, done, info = env.step(action)

            episode_reward += reward
            
            action_next = AI_player.Get_Action(observation = observation_next)
            train_loss= AI_player.Q_FC_Train(observation = observation, action = action, reward = reward, done = done, observation_next = observation_next, action_next = action_next)

            t += 1
            action = action_next
            observation = observation_next

            if done:
                
                # Save episode rewards
                episode_rewards.append(episode_reward)
                
                # Save the best model
                episode_rewards_average = np.average(episode_rewards[-EVALUATION_INTERVAL:])
                if (episode > EVALUATION_INTERVAL) and (episode_rewards_average > episode_reward_average_max):
                    episode_reward_average_max = episode_rewards_average
                    AI_player.Q_FC_Save(model_name = 'AI_Sarsa_Best')
                    
                # Print episode information
                print("#" * 50)
                print("Episode: %d" % episode)
                print("Episode reward: %f" % episode_reward)
                print("Episode reward average max: %f" % episode_reward_average_max)
                print("Episode finished after {} timesteps".format(t+1))
                print("Train Loss: %f" % train_loss)
                                
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
    AI_player = OpenAI_Sarsa_FC(num_actions = num_actions, num_features = num_features)
    AI_player.Q_FC_Restore(model_name = 'AI_Sarsa_Best')
    
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
