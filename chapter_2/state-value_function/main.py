import numpy as np

from random import randrange

SQUARES = 5
STARTING_POSITION = 0
GOAL_SQUARE = SQUARES - 1

# random policy (strategy)
def policy():
   return randrange(2)

def perform_action(position):
   return move(position, policy())

# 0 = left
# otherwise right
def move(position, direction):
   return position + (-1 if direction == 0 else 1)

# Reward is only given at the end (when won or died)
def get_reward(died):
   return 0 if died else SQUARES

def start(epochs):
   # Discounted reward
   G = [0] * GOAL_SQUARE
   counts = [0] * GOAL_SQUARE
   
   for _ in range(0, epochs):
      position = STARTING_POSITION
      states = [position]

      while True:
         position = perform_action(position)
         is_terminal_state = position == -1 or position == GOAL_SQUARE
      
         if is_terminal_state:
            died = position == -1
            reward = get_reward(died)

            for state in states:
               counts[state] += 1
               G[state] += reward
            
            break

         states.append(position)

   print(f'Finished the simulation. Expected returns (state-value function)={np.divide(G, counts, dtype="f")}')

start(10000)