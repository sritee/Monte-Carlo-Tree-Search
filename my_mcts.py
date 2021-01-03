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

np.random.seed(1)

def combinations(space):
    if isinstance(space, gym.spaces.Discrete):
        return list(range(space.n))
    elif isinstance(space, gym.spaces.Tuple):
        return itertools.product(*[combinations(s) for s in space.spaces])
    else:
        raise NotImplementedError


class MCTSNode:

    C = 1 # UCB weighing term. Increasing it results in exploring for longer time.
    def __init__(self, parent, action):

        self.parent = parent
        self.action = action  # action taken to get to this from the parent
        self.children = []
        self.visits = 0
        self.value = 0


    @property
    def ucb_score(self):
        if self.visits == 0:
            return 10000 + np.random.rand(1).item()  # arbitrarly large plus noise for tie-break
        return self.value + self.C * np.sqrt(np.log(self.parent.visits)/self.visits)

    @property
    def best_child(self):
        """return child with highest value. if no visited child return None"""
        visited_children = [child for child in self.children if child.visits > 0]
        if not visited_children:  # none of them visited
            return None
        return max(visited_children, key=lambda child : child.value)

    def update_value(self, monte_carlo_return):

        # running average update
        self.visits += 1
        self.value += 1/(self.visits) * (monte_carlo_return - self.value)


class MCTS:


    def __init__(self, gamma = 0.99):

        self.gamma = gamma
        self.root = MCTSNode(None, None) # root has no parent

    def get_plan_from_root(self):
        """
        from self.root, find the sequence of actions by accessing node.best_child
        i.e greedy policy w.r.t value function
        """

        root = self.root
        plan = []
        while root.best_child:
            plan.append(root.best_child.action)
            root = root.best_child

        return plan

    def update_tree(self, action):
        """
        update the tree with an action. this means an action has been
        executed in the real-world, now the mcts root node will change
        to one of its children so we can re-use the mcts tree.
        """
        child_actions = [child.action for child in self.root.children]
        if action in child_actions:
            self.root = self.root.children[child_actions.index(action)]
            self.root.parent = None
        else:
            self.root = MCTSNode(None, None)

    @staticmethod
    def evaluate_mcts_policy(root_node, env, max_depth=500, render=True):

        """
        greedy evaluation of mcts policy
        args:
            root_node: the root noded of mcts tree
            env: the openai gym environment
            max_depth: maximum depth to simulate
            render: whether to render the evaluation
        """

        cum_reward = 0
        env_backup = copy.deepcopy(env)
        done = False
        depth = 0

        while root_node and root_node.best_child and not done and depth < max_depth:

            best_action = root_node.best_child.action
            _, r, done, info = env_backup.step(best_action)
            cum_reward += r
            depth += 1
            root_node = root_node.best_child
            if render:
                env_backup.render()

        while not done and depth < max_depth: # entering unexplored region, take random actions
            random_action = env_backup.action_space.sample()
            _, r, done, _ = env_backup.step(random_action)
            cum_reward += r
            depth+= 1
            if render:
                env_backup.render()

        env_backup.close()
        print(f'planner achieved cumulative return of {cum_reward}')


    def compute_plan(self, env: gym.Env, num_iterations=10000, max_depth=500):

        """
        compute a MCTS plan by executing num_iterations rollouts with max_depth.
        returns list of actions found by MCTS (open-loop control sequence)
        """

        for iteration in range(num_iterations):

            node = self.root
            env_mcts = copy.deepcopy(env)
            depth = 0
            # if iteration % 100 == 0:
            #     print(f'performing iteration {iteration} of MCTS')

            done = False
            trajectory_rewards = []  # store the individual reward along the trajectory.

            while node.children:
                node = max(node.children, key=lambda node: node.ucb_score)

                _, reward, done, info = env_mcts.step(node.action)  # assume deterministic environment
                trajectory_rewards.append(reward)
                depth+=1
                if done or depth >= max_depth:
                    break

            # at this point we are either at a leaf node or at a terminal state
            if not done and depth < max_depth: # it's a leaf node. let's add its children
                node.children = [MCTSNode(node, a) for a in combinations(env_mcts.action_space)]

                # rollout with a random policy till we reach a terminal state
                leaf_rollout_return = 0
                leaf_rollout_depth = 0

                while not done and depth < max_depth:
                    _, reward, done, _ = env_mcts.step(env_mcts.action_space.sample())

                    leaf_rollout_return += self.gamma ** leaf_rollout_depth * reward  # discounted
                    leaf_rollout_depth += 1
                    depth+=1

                # append the Monte carlo return of the leaf node to trajectory reward.
                trajectory_rewards.append(leaf_rollout_return)

            # start from end of trajectory back till root
            reward_idx = len(trajectory_rewards) - 1
            discounted_return = 0
            # return of node is rewards from that node till leaf node,
            # plus return of leaf node, adjusted by the discount factor
            while node:  # backup the Monte carlo return till root

                discounted_return = self.gamma * discounted_return + trajectory_rewards[reward_idx]
                node.update_value(discounted_return)

                # move to parent node
                reward_idx -= 1
                node = node.parent

        return self.get_plan_from_root()




if __name__ == '__main__':

    env_name = 'CartPole-v1'
    gamma = 0.99  # discount factor
    replanning_horizon = 50 # length of open-loop control sequence after which MCTS replans
    max_search_depth = 300  # maximum tree depth, depending on time-scale of environment.
    num_iterations = 500   # number of iterations of MCTS tree updates per planning request.

    env = gym.make(env_name)
    # seed the environment. results still not exactly reproducible since rollout policy stochastic.
    # seeding the action space would result in issues since rollout sequence will be fixed due to deepcopy.
    env.seed(2)
    env.reset()

    agent = MCTS(gamma=gamma)
    cum_reward = 0

    done = False
    env_orig = copy.deepcopy(env)  # make a copy of env to render whole sequence at end
    action_sequence = []

    while not done:

        print(f'requesting MCTS planning')
        plan = agent.compute_plan(env, num_iterations=num_iterations, max_depth=max_search_depth)
        actions = plan[0:replanning_horizon] # get the sequence of actions to execute
        print(f'executing open-loop sequence for {len(actions)} timesteps and updating MCTS state for replanning')
        for action in actions:
            _, reward, done, _ = env.step(action)
            action_sequence.append(action)
            agent.update_tree(action) # update the internal state of MCTS agent with action executed
            cum_reward += reward
            if done:
                break

    print(f'environment episode had return of {cum_reward}. Rendering executed policy.')
    done = False
    for action in action_sequence:
        _, reward, done, _ = env_orig.step(action)
        env_orig.render()
        if done:
            break

