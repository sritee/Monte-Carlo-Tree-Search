# MCTS
Monte Carlo Tree Search implementation for a receding horizon control setting.

# Samples

<p align="center">
  <img src="/samples/gym_animation.gif" alt="Sublime's custom image"/>
</p>
<p align="center"> 
   CartPole-v1
 </p>
<p align="center">
  <img src="/samples/intersection.gif" alt="Sublime's custom image"/>
</p>

<p align="center"> 
   intersection-v0 from https://github.com/eleurent/highway-env
 </p>

## Features

- Supports sparse/dense rewards
- Supports replanning for receding horizon control.
- Simple and Stand-alone. 

## Tested on

- CartPole-v1, FrozenLake8x8, intersection-v0 (from highway_env), Taxi-v0

## Hyperparameters
- gamma (discount factor): 0.99 should work across most. 
- replanning_horizon: number of timesteps after which MCTS is queried again for a new plan. It is set based on how long a sucessful episode could last and
  granularity of control required. For example, in FrozenLake8x8-v0, a replannig horizon of 5 could be appropriate, but in cartpole, a horizon of 50 works fine.
- max_tree_depth: maximum depth to which the tree is expanded. needs to be finite to support continuous control tasks. 
- num_iterations: number of rollouts performed per MCTS query.

## TODO 

- Reset tree statistics/visits on stepping the tree.
- Parallelize implementation.

## References

- https://gist.github.com/blole/dfebbec182e6b72ec16b66cc7e331110?fbclid=IwAR2VGY2HxI6sKhqsO2FPjP27qW44gsttzmuOn0IKNo2rabD6s3F9r6vJc0Q
- https://github.com/eleurent/rl-agents/blob/master/rl_agents/agents/tree_search/mcts.py
