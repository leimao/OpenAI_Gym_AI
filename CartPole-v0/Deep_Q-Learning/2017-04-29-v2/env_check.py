import gym

ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)

print("env.action_space")
print(env.action_space)
print("env.observation_space")
print(env.observation_space)
print("env.observation_space.high")
print(env.observation_space.high)
print("env.observation_space.low")
print(env.observation_space.low)

