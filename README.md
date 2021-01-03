# MCTS
Monte Carlo Tree Search implementation for a receding horizon control setting.


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
