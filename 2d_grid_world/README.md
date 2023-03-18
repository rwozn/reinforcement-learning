# Environment specification:

## 2D grid world with NxM tiles (N=10, M=4).

## The objective is to move to the goal tile (at (9, 0)).

- State space: [0, 1, 2, ..., N-1]x[0, 1, 2, ..., M-1] tiles

- Action space: [0 = move left, 1 = move right, 2 = move up, 3 = move down]

- Terminal state:
    - state = goal_tile (win)
    - state ∈ cliff_tiles (lose)

- Reward:
    - p when won (p = 0)
    - q when lost (q = -100)
    - r otherwise (r = -1)
