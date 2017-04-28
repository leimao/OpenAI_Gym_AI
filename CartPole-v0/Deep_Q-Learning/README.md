# Deep Q-Learning CartPole AI

Author: Lei Mao

Date: 4/28/2017

## Introduction

[CartPole-v0](https://gym.openai.com/envs/CartPole-v0) is one of the learning environments in OpenAI Gym. Here I used Deep Q-Learning to learn how to play CartPole.

Before Training:

![](presentation/episode_0.gif)

After Training:

![](presentation/episode_27000.gif)

## Installation Dependence

* Python 2.7
* Tensorflow 1.0
* Keras 2.0.3
* OpenAI Gym Beta

## AI Features

The CartPole AI was developed using Deep Q-Learning Neural Network. 

The AI program was written in Python using Keras, which makes the code for neural network neat and easy to understand.

The AI was wrapped as class, which makes it universal for all different kind of environment APIs in OpenAI Gym.

## Run AI in CartPole

The AI needs to be trained before the game. To train the AI, run the command "python OpenAI.py -m train". The AI has already been trained and stored as AI_model.h5 file if you do not want to do the training.

To allow the trained AI to play the game, run the command "python OpenAI.py -m test".

## Optimazation Notes

<https://leimao.github.io/journal/OpenAI-Gym-CartPole.html>

## AI Performance

<https://gym.openai.com/evaluations/eval_ewr0DWHeTmGE6x1NGQ1LiQ>
