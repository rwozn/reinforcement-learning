import numpy as np

from random import randrange

SQUARES = 5
STARTING_POSITION = 0
GOAL_SQUARE = SQUARES - 1

# random policy (strategy)
def policy():
   return randrange(2)

ACTION_LEFT = 0
ACTION_RIGHT = 1

def move(position, direction):
   return position + (-1 if direction == ACTION_LEFT else 1)

# Reward is only given at the end (when won or died)
def get_reward(died):
   return 0 if died else SQUARES

NUM_ACTIONS = 2

def start(epochs, debug=False):
   # Discounted reward
   G = [[STARTING_POSITION] * GOAL_SQUARE for _ in range(NUM_ACTIONS)]

   counts = [[0] * GOAL_SQUARE for _ in range(NUM_ACTIONS)]

   for epoch in range(0, epochs):
      position = STARTING_POSITION
      states = [position]
      actions = []

      while True:
         action = policy()
         position = move(position, action)

         is_terminal_state = position == -1 or position == GOAL_SQUARE
      
         if is_terminal_state:
            actions.append(action)

            died = position == -1
            reward = get_reward(died)

            for i in range(len(states)):
               state = states[i]
               action = actions[i]

               counts[action][state] += 1
               G[action][state] += reward
            
            if debug:
               print(f'it={epoch + 1}. {"Died" if died else "Won"}. Actions={actions}, States={states}, Reward={reward}, G=\n{G}, Counts=\n{counts}')
            
            break

         actions.append(action)
         states.append(position)

   Q_function = np.divide(G, counts, dtype="f")

   print(f'Finished the simulation. Expected returns (action-value (Q) function)=\n{Q_function}')

   print(f"Optimal policy: {np.max(Q_function, axis=0)} (actions={np.argmax(Q_function, axis=0)})")

start(10000)