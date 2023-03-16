Environment specification:

1D grid world with N tiles (N=5).

The goal is to move to the right-most tile.

State: N tiles

Action space: [0 = move left, 1 = move right]

Terminal state:
    - action = 0 when state = 0 (move to the left when on the first tile)
    - action = 1 when state = N-1 (step onto the right-most tile)

Reward is given when the agent either wins (reward = N) or dies (reward = 0).