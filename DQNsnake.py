# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 22:49:39 2020

@author: Nichita Vatamaniuc
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import matplotlib.pyplot as plt

import gym
import sneks

import datetime
import random
import collections


def parameters_defenition():
    parameters = dict()
    parameters['epsilon_decay'] = 1/90
    parameters['learning_rate'] = 0.0005
    parameters['first_layer_size'] = 200
    parameters['second_layer_size'] = 100
    parameters['third_layer_size'] = 50
    parameters['output_dim'] = 4
    parameters['output_activation'] = 'sigmoid'
    parameters['episodes_to_play'] = 150
    parameters['memory_size'] = 2500
    parameters['memory_batch'] = 500
    parameters['train'] = True
    parameters['environment'] = 'hungrysnek-raw-16-v1'
    parameters['render'] = False
    parameters['weights_name'] = 'weights' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.hdf5'
    parameters['console_log'] = True
    return parameters


def draw_graph(reward, name):
    plt.plot(np.asarray(reward))
    plt.title(name)
    plt.show() 


class DQNAgent(object):
    def __init__(self, parameters):
        self.reward = 0
        self.gamma = 0.9
        self.short_memory = np.array([])
        self.epsilon = 1
        self.learning_rate = parameters['learning_rate']
        self.first_layer_size = parameters['first_layer_size']
        self.second_layer_size =  parameters['second_layer_size']
        self.third_layer_size = parameters['third_layer_size']
        self.long_memory = collections.deque(maxlen = parameters['memory_size'])
        self.model = self.neural_network()
        
        
    def neural_network(self):
        model = Sequential()
        model.add(Dense(self.first_layer_size, activation = 'relu', input_dim = 6))
        model.add(Dense(self.second_layer_size, activation = 'relu'))
        model.add(Dense(self.third_layer_size, activation = 'relu'))
        model.add(Dense(parameters['output_dim'], activation = parameters['output_activation']))
        model_optimizer = Adam(self.learning_rate, amsgrad = True)
        model.compile(loss = 'mse', optimizer = model_optimizer)
        return model
    
    
    @staticmethod
    def preprocess(state):
            
        #Finding of the head of snake
        head = np.where(state == 101)
        head = np.asarray(head)
        head = head.squeeze()
    
        # vision is 4 booleans that tell if in left, right, upward, downward is a barrier
        vision = []
        if state[head[0]-1][head[1]] != 0.0 and state[head[0]-1][head[1]] != 64.0:
            vision.append(1)
        else: vision.append(0)
        if state[head[0]+1][head[1]] != 0.0 and state[head[0]+1][head[1]] != 64.0:
            vision.append(1)
        else: vision.append(0)
        if state[head[0]][head[1]-1] != 0.0 and state[head[0]][head[1]-1] != 64.0:
            vision.append(1)
        else: vision.append(0)
        if state[head[0]][head[1]+1] != 0.0 and state[head[0]][head[1]+1] != 64.0:
            vision.append(1)
        else: vision.append(0)
    
      
        #Finding relative position of food to snake's head
        food = np.where(state == 64)
        food = np.asarray(food)
        food = food.squeeze()
        
        rel_food_pos_x = head[0] - food[0]
        rel_food_pos_y = head[1] - food[1]
        
        #Adding relative pos of food to vision
        vision.append(rel_food_pos_x)
        vision.append(rel_food_pos_y)
        
        return tf.reshape(np.asarray(vision),(1,6))
        

    def update_long_memory(self, state, action, reward, next_state, done):
        self.long_memory.append((state, action, reward, next_state, done))
    
    
    def train_long_memory(self, memory_to_train, memory_batch):
        if len(memory_to_train) > memory_batch:
            batch_to_train = random.sample(memory_to_train, memory_batch)
        else:
            batch_to_train = memory_to_train
        for state, action, reward, next_state, done in batch_to_train:
            target = reward
            input_state = self.preprocess(state)
            if not done:
                input_next_state = self.preprocess(next_state)
                target = reward + self.gamma * np.amax(self.model.predict(input_next_state))
            target_expected = self.model.predict(input_state)
            target_expected[0][np.argmax(action)] = target
            self.model.fit(input_state, target_expected, epochs = 1, verbose = 0)
            
          
    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        input_state = self.preprocess(state)
        if not done:
            input_next_state = self.preprocess(next_state)
            target = reward + self.gamma * np.amax(self.model.predict(input_next_state))
        target_expected = self.model.predict(input_state)
        target_expected[0][np.argmax(action)] = target
        self.model.fit(input_state, target_expected, epochs = 1, verbose = 0)
        
        
    def test_runs(self, tests, env):
        for runs in range(tests):
            observation = env.reset()
            done = False
            rewards = []
            state_new, reward, done, _ = env.step(np.argmax(self.model.predict(self.preprocess(observation))))
            while not done:
                state_new, reward, done, _ = env.step(np.argmax(self.model.predict(self.preprocess(state_new))))
                env.render()
                rewards.append(reward)
            print('Total episode reward: ' + str(sum(np.asarray(rewards))))    
                
        
        
def train_model(parameters):
    env = gym.make(parameters['environment'])
    
    agent = DQNAgent(parameters)
    if parameters['train']:
        #initialization
        episodes_played = 0
        total_train_reward = []
        
        while episodes_played < parameters['episodes_to_play']:
            steps = 0
            episode_reward = []
            done = False
            env.reset()
            while not done:
                agent.epsilon = 1 - (episodes_played * parameters['epsilon_decay'])
                if agent.epsilon < 0:
                    agent.epsilon = 0
                    
                state_old = env._get_state()
                input_state_old = agent.preprocess(state_old)
                

                if np.random.random() < agent.epsilon:
                    action_to_do = tf.keras.utils.to_categorical(np.random.randint(0,3), num_classes = 4)
                else:
                    prediction = agent.model.predict(input_state_old)
                    action_to_do = tf.keras.utils.to_categorical(np.argmax(prediction), num_classes = 4)
                    
                next_state, reward, done , _ = env.step(np.ndarray.tolist(action_to_do).index(np.amax(action_to_do)))
                episode_reward.append(reward)
                
                if parameters['render']:
                    env.render()
                
                agent.train_short_memory(state_old, action_to_do, reward, next_state, done)
                agent.update_long_memory(state_old, action_to_do, reward, next_state, done)
                steps += 1
            
            episodes_played += 1    
            agent.train_long_memory(agent.long_memory, parameters['memory_batch'])
            total_train_reward.append(sum(episode_reward))
            
            if parameters['console_log']:
                 print('Epsilon: ' + str(agent.epsilon) + ' EpReward: ' + str(sum(np.asarray(episode_reward))) + ' Episode: ' + str(episodes_played) + ' Steps: ' + str(steps))
                 
        agent.test_runs(10, env)
        draw_graph(total_train_reward, parameters['output_activation'])
        agent.model.save_weights(parameters['weights_name']) 
        print('Model was saved as ' + parameters['weights_name'])
        env.close()                
         
    else:
        print('Print weights name (should be in the same directory as script): ')
        weights_name = input()
        agent.model.load_weights(weights_name)
        agent.test_runs(10, env)
        env.close() 
             
    
        
if __name__ == '__main__':
    parameters = parameters_defenition()
    train_model(parameters)
    
            
