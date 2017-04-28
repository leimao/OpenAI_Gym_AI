# OpenAI Gym AI Collections

Author: Lei Mao

Date: 4/28/2017

## Introduction

[OpenAI Gym](https://gym.openai.com/) is a platform where you could test your intelligent learning algorithm in various application, including games and virtual physics experiments. It provides APIs for all these applications for the convenience of integrating the algorithms into the application. The API is called "environment" in OpenAI Gym. On one hand, the environment only receives "action" instructions as input and outputs the observation, reward, signal of termination, and other information. On the other hand, your learning algorithm receives observation(s), reward(s), signal(s) of termination as input and outputs the action. So in principle, one can develop a learning algorithm and wrapped it into a class object. It could test all the enviroments in OpenAI Gym.
Before Training:

Here I developed several AIs to play in the OpenAI Gym environments using multiple different algorithms. The parameters for these algorithms have been carefully tuned to reach good AI performance.

## Installation Dependences

* Python 2.7
* Tensorflow 1.0
* Keras 2.0.3
* OpenAI Gym Beta
* OpenCV 2.4.13
* Pygame 1.9.3

## How to train and test AI

Please go to the environment folders for detailed instructions.