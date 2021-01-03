# MCTS
Monte Carlo Tree Search implementation for a receding horizon control setting.

# Samples

<p align="center">
  <img src="/samples/gym_animation.gif" alt="Sublime's custom image"/>
</p>
<p align="center">
  <img src="/samples/intersection.gif" alt="Sublime's custom image"/>
</p>

## Features

- Supports sparse/dense rewards
- Supports replanning for receding horizon control.
- Simple and Stand-alone. 

## Tested on

- CartPole-v1, FrozenLake8x8, intersection-v0 (from highway_env), Taxi-v0

## TODO 

- Reset tree statistics/visits on stepping the tree.
- Parallelize implementation.

## References

- https://gist.github.com/blole/dfebbec182e6b72ec16b66cc7e331110?fbclid=IwAR2VGY2HxI6sKhqsO2FPjP27qW44gsttzmuOn0IKNo2rabD6s3F9r6vJc0Q
- https://github.com/eleurent/rl-agents/blob/master/rl_agents/agents/tree_search/mcts.py
