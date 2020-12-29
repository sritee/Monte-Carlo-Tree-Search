#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 10:08:50 2020
inspired by 
https://gist.github.com/blole/dfebbec182e6b72ec16b66cc7e331110?fbclid=IwAR2VGY2HxI6sKhqsO2FPjP27qW44gsttzmuOn0IKNo2rabD6s3F9r6vJc0Q
"""

import gym
import numpy as np
import copy
import itertools
import random
import time

def combinations(space):
    if isinstance(space, gym.spaces.Discrete):
        return list(range(space.n))
    elif isinstance(space, gym.spaces.Tuple):
        return itertools.product(*[combinations(s) for s in space.spaces])
    else:
        raise NotImplementedError
        
class MCTSNode:
    
    C = 1  # UCB weighing term
    def __init__(self, parent, action):
        
        self.parent = parent
        self.action = action  # action taken to get to this from the parent
        self.children = []
        self.visits = 0
        self.cum_value = 0
        
        
    @property
    def value(self):
        if self.visits == 0:
            return 10000 # arbitrarily large
        else:
            return self.cum_value / self.visits
      
    @property
    def ucb_score(self):
        if self.visits == 0:
            return 100000 # arbitrarily large
        else:
            assert self.parent is not None
            return self.value + self.C * np.sqrt(np.log(self.parent.visits)/self.visits)

    @property
    def best_child(self):
        if not self.children:
            return None
        return max(self.children, key=lambda node: node.value)


def evaluate_mcts_policy(root_node, env, render=True):
    """greedy evaluation of mcts policy"""
    
    best_actions = []
    cum_reward = 0
    env_backup = copy.deepcopy(env)
    done = False
    
    while root_node and root_node.best_child and not done:
        
        best_action = root_node.best_child.action
        _, r, done, info = env_backup.step(best_action)
        cum_reward += r
        best_actions.append(best_action)
        root_node = root_node.best_child
        if render:
            env_backup.render()
            time.sleep(1)
            
    while not done: # entering unexplored region, take random actions
        random_action = env_backup.action_space.sample()
        _, r, done, _ = env_backup.step(random_action)
        cum_reward += r
        best_actions.append(random_action)
    
    env_backup.close()
    print(f'best action of length {len(best_actions)} and return {cum_reward}')

def mcts_policy(env, num_iterations = 10000):
    
    root = MCTSNode(None, None)
    # do several iterations of the following

    # if has children
    #   while node.children is not None
            # node = parent.best child
            # keep track of sequence of actions or step the simulator
    
    # when this statement is reached, we are at a node = leaf node
    # add its children with cur node as parent.
    # do a rollout from this node, for this, we'd need simulator with state equals this node at every step
    # from start node, step the environment with sequences of actions taken to reach here
    # now we have the monte carlo value of this node, need to back it up to parents
    
    # while node is not None:
        # node.cum_value += monte_carlo_return
        # node.visits += 1
        # node = node.parent
        
    
    # def evaluate tree which is greedy, and picks max value child
    # if no child, it just samples randomly after that till episode termination
    
    
    for iteration in range(num_iterations):
        
        node = root
        env_mcts = copy.deepcopy(env)
        if iteration % 500 == 0:
            print(f'performing iteration {iteration} of MCTS')
        
        done = False
        
        while node.children:
            node = max(node.children, key=lambda node: node.ucb_score)
            _, reward, done, info = env_mcts.step(node.action)  # assume deterministic environment
            # if reward == 1:
            #     print('found')
            if done:
                break
        
        # we are either at a leaf node or terminal state
        if not done:
            # leaf node
            node.children = [MCTSNode(node, a) for a in combinations(env_mcts.action_space)]
            leaf_val = 0
        else:
            # terminal state. in this case the MC return of this state is just the last reward
            leaf_val = reward
            # rollout with a random policy till we reach a terminal state
            while not done:
                _, reward, done, _ = env_mcts.step(env_mcts.action_space.sample())
                # if reward == 1:
                #     print('found')
                leaf_val += reward

        while node:  # backup the MC return to parent nodes
            node.cum_value += leaf_val
            node.visits += 1
            node = node.parent
        
        if  iteration % 500 == 0:
            evaluate_mcts_policy(root, env, render=False)


# Note the code makes sense in the case of a sparse reward at the end
# other the monte carlo returns are not propogated properly
# doesn't seem to work on pong, random policy is not good enough to explore by the looks of it
#env = gym.make('LunarLander-v2')
#env = gym.make('CartPole-v1')
#env = gym.make('Taxi-v3')
env = gym.make('FrozenLake-v0')
#env = gym.make('Pong-v4')
#env = gym.make('Acrobot-v1')



trials = 10
for _ in range(trials):
    env.reset()
    mcts_policy(env, num_iterations = 10000)
    
# for _ in range(200):
#     _, r, d , i = env.step(env.action_space.sample())
#     env.render()
#     time.sleep(0.1)
#     print(r, d)
    
env.close()



