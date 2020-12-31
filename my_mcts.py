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

def combinations(space):
    if isinstance(space, gym.spaces.Discrete):
        return list(range(space.n))
    elif isinstance(space, gym.spaces.Tuple):
        return itertools.product(*[combinations(s) for s in space.spaces])
    else:
        raise NotImplementedError
        
class MCTSNode:
    
    C = 1  # UCB weighing term. Increasing it results in exploring for longer time.
    def __init__(self, parent, action):
        
        self.parent = parent
        self.action = action  # action taken to get to this from the parent
        self.children = []
        self.visits = 0
        self.cum_value = 0
        
        
    @property
    def value(self):
        assert self.visits != 0, 'unvisited node has no value'
        return self.cum_value / self.visits
      
    @property
    def ucb_score(self):
        if self.visits == 0:
            return 10000 + np.random.rand(1)  # arbitrarly large plus noise for tie-break
        return self.value + self.C * np.sqrt(np.log(self.parent.visits)/self.visits)

    @property
    def best_child(self):
        """return child with highest value. if no visited child return None"""
        visited_children = [child for child in self.children if child.visits > 0]
        if not visited_children:  # none of them visited
            return None
        return max(visited_children, key=lambda child : child.value)


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
            
    while not done: # entering unexplored region, take random actions
        random_action = env_backup.action_space.sample()
        _, r, done, _ = env_backup.step(random_action)
        cum_reward += r
        best_actions.append(random_action)
    
    env_backup.close()
    print(f'best action of length {len(best_actions)} and return {cum_reward}')

def mcts_policy(env, num_iterations = 10000):
    
    root = MCTSNode(None, None) # root has no parent
    
    for iteration in range(num_iterations):
        
        node = root
        env_mcts = copy.deepcopy(env)

        if iteration % 100 == 0:
            print(f'performing iteration {iteration} of MCTS')
        
        done = False
        tree_rewards = []
        
        while node.children:
            node = max(node.children, key=lambda node: node.ucb_score)
            _, reward, done, info = env_mcts.step(node.action)  # assume deterministic environment
            tree_rewards.append(reward)
            if done:
                break
        
        # we are either at a leaf node or at a terminal state
        if not done: # leaf node. let's add its children
            node.children = [MCTSNode(node, a) for a in combinations(env_mcts.action_space)]
            leaf_value = 0
        else:
            # terminal state. in this case the Monte carlo return of this state is just the last reward
            leaf_value = reward
        
        # rollout with a random policy till we reach a terminal state
        while not done:
            _, reward, done, _ = env_mcts.step(env_mcts.action_space.sample())
            leaf_value += reward
        
        # for the final leaf node, the monte carlo return is just the leaf value
        node.cum_value += leaf_value
        node.visits += 1
        node = node.parent
        
        # for other nodes, monte carlo return is sum of rewards from that node + leaf node monte carlo return
        acum_rewards = np.flip(np.cumsum(tree_rewards))
        return_idx = -1 # indicates index of cumulative return we are interested in 

        while node:  # backup the MC return to parent nodes
            # sum of tree rewards from that node + leaf_val
            node.cum_value  += leaf_value + acum_rewards[return_idx]
            return_idx-= 1
            node.visits += 1
            node = node.parent
        
        if  iteration % 500 == 0:
            evaluate_mcts_policy(root, env, render=True)



# working
env = gym.make('CartPole-v1')
#env = gym.make('FrozenLake-v0')
# maybe working
#env = gym.make('Taxi-v3')
# not working
#env = gym.make('Pong-v0')
#env = gym.make('Acrobot-v1')
#env = gym.make("intersection-v0")


trials = 1
for _ in range(trials):
    env.reset()
    mcts_policy(env, num_iterations = 10000)
    
env.close()



