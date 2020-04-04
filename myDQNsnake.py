#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 22:49:39 2020

@author: Nichita Vatamaniuc
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random
import numpy as np
#from operator import add
import collections

import gym
import sneks


def define_parameters():
    params = dict()
    params['epsilon_decay_linear'] = 1/75
    params['learning_rate'] = 0.0005
    params['first_layer_size'] = 512   # neurons in the first layer
    params['second_layer_size'] = 256   # neurons in the second layer
    params['third_layer_size'] = 128    # neurons in the third layer
    params['episodes'] = 20            
    params['memory_size'] = 2500
    params['batch_size'] = 500
    #params['weights_path'] = 'weights/weights.hdf5'
    #params['load_weights'] = True
    params['train'] = True
    params['env'] = 'snek-raw-16-v1' #snek-rgb-16-v1 babysnek-raw-16-v1
    return params

class DQNAgent(object):
    def __init__(self, params):
        self.reward = 0
        self.gamma = 0.9
        #self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        #self.agent_target = 1
        #self.agent_predict = 1
        self.learning_rate = params['learning_rate']        
        self.epsilon = 1
        #self.actual = []
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.memory = collections.deque(maxlen=params['memory_size'])
        #self.weights = params['weights_path']
        #self.load_weights = params['load_weights']
        self.model = self.network()

    def network(self):
        model = Sequential()
        model.add(Dense(self.first_layer, activation='relu', input_shape=(256,)))
        model.add(Dense(self.second_layer, activation='relu'))
        model.add(Dense(self.third_layer, activation='relu'))
        model.add(Dense(4, activation='linear'))
        opt = Adam(self.learning_rate, amsgrad=False)
        model.compile(loss='mse', optimizer=opt)
        return model

    def remember(self, state, action, reward, next_state, done):      
        self.memory.append((state, action, reward, next_state, done))
        
        
    def replay_new(self, memory, batch_size):
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state.flatten().reshape(1,256)))
            target_f = self.model.predict(np.array(state.flatten().reshape(1,256)))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array(state.flatten().reshape(1,256)), target_f, epochs=1, verbose=0)
            
            
    '''def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.flatten()) #next_state.reshape((1, 11))
        target_f = self.model.predict(next_state.flatten())
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 11)), target_f, epochs=1, verbose=0)'''
        
    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(np.asarray(next_state.flatten().reshape(1,256)))[0])
        target_f = self.model.predict(np.asarray(state.flatten().reshape(1,256)))
        #print(target_f)
        target_f[0][np.argmax(action)] = target
        #print(target_f)
        self.model.fit(state.flatten().reshape(1,256), target_f, epochs=1, verbose=0)



def run(params):
    agent = DQNAgent(params)
    
    env = gym.make(params['env'])
    
    #initialize
    env.reset()
    done = False
    episode = 0
    rewards = []
    steps = 0
    
    while episode < params['episodes']:
        env.reset()
        done = False
        while not done:
            steps = steps + 1
            if not params['train']:
                agent.epsilon = 0
            else:
                agent.epsilon = 1 - (episode * params['epsilon_decay_linear']) #params['epsilon_decay_linear']
            
            if agent.epsilon < 0:
                agent.epsilon = 0
            
            #print(agent.epsilon)
            #get old state
            state_old = env._get_state()
           #env.render()
            
            #perform random action based on agent.epsilon, or choose the action from NN
            if np.random.randint(0,1) < agent.epsilon: # random int??
                #final_move = np.random.randint(0,3) #tf.keras.utils.to_categorical(np.random.randint(0,3)) #num_classes=3
                final_move = tf.keras.utils.to_categorical(np.random.randint(0,2), num_classes=4)
            else:
                #predict
                prediction = agent.model.predict(state_old.flatten().reshape(1,256)) #reshape?
                #prediction = agent.model.predict(state_old.reshape(1,16,16))
                #final_move = np.argmax(prediction, axis=1)[0] #tf.keras.utils.to_categorical(np.argmax(prediction[0]), num_classes=3) # ???
                final_move = tf.keras.utils.to_categorical(np.argmax(prediction[0]), num_classes=4)
        
            #perform new move and agent new state
            state_new, reward, done, _ = env.step(np.ndarray.tolist(final_move).index(np.amax(final_move)))
            
            #env.render()
            rewards.append(reward)
            #env.render()
            if params['train']:
                #train short memory
                agent.train_short_memory(state_old, final_move, reward, state_new, done)
                agent.remember(state_old, final_move, reward, state_new, done)
            
            if steps % 20 == 0:
                awg_rwd = np.asarray(rewards)
                print('Awg reward: ' + str(awg_rwd[max(0,steps-100):(steps+1)].mean()))
            
            #record
        episode = episode + 1    
        print('episode: ' + str(episode))        
        if params['train']:
            agent.replay_new(agent.memory, params['batch_size'])
            
    for i in range(10):
        obs = env.reset()
        done = False
        while not done:
            state_new, reward, done, _ = env.step(np.argmax(agent.model.predict(np.asarray(obs.flatten().reshape(1,256)))))
            env.render()
        print(reward)    
            


if __name__ == '__main__':
    params = define_parameters()
    run(params)        
        
                   
