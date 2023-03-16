import random

import numpy as np

SQUARES = 5
STARTING_POSITION = 0
GOAL_SQUARE = SQUARES - 1

ACTION_LEFT = 0
ACTION_RIGHT = 1

def move(position, direction):
   return position + (-1 if direction == ACTION_LEFT else 1)

# Reward is only given at the end (when won or died)
def get_reward(died):
   return 0 if died else SQUARES

NUM_STATES = GOAL_SQUARE
NUM_ACTIONS = 2

def generate_trajectory(policy):
   position = STARTING_POSITION

   states = [position]
   actions = []

   while True:
      action = policy(position)
      position = move(position, action)

      is_terminal_state = position == -1 or position == GOAL_SQUARE
   
      if is_terminal_state:
         break
      
      actions.append(action)
      states.append(position)

   died = position == -1

   return list(zip(states, actions + [action], [0] * len(actions) + [get_reward(died)]))

"""
Monte Carlo techniques solve Markov decision processes (MDPs) by sampling their way to an
estimation of a value function. They can do this without explicit knowledge of the transition
probabilities and can efficiently sample large state spaces. But they need to run for an
entire episode before the agent can update the policy.
"""
# discounted return
# gamma - discount rate/factor - the lower the faster it ignores future rewards: 0 <= gamma <= 1
# (usually between 0.9 a 0.99)
# 0 => only present reward is taken into consideration
# 1 => discounted return (G) - sum of future rewards
def onpolicy_monte_carlo(epochs, policy, gamma=1):
   Q = [[0] * NUM_STATES for _ in range(NUM_ACTIONS)]

   returns = [[[] for _ in range(NUM_STATES)] for _ in range(NUM_ACTIONS)]

   for _ in range(epochs):
      G = 0
      
      trajectory = generate_trajectory(lambda state: policy(state, Q))
      trajectory = list(reversed(trajectory))

      for i, (state, action, reward) in enumerate(trajectory):
         G = gamma * G + reward

         not_handled = len([j for j, episode in enumerate(trajectory[:i]) if episode[0] == state and episode[1] == action]) == 0

         if not_handled:
            current_return = returns[action][state]
            current_return.append(G)

            Q[action][state] = np.mean(current_return)

   return Q

# epsilon-greedy search
# epsilon - exploration probability - the higher epsilon, the more the agent explores: 0 <= epsilon <= 1
epsilon = 0.8

def get_best_action(Q, state):
   return np.argmax([Q[action][state] for action in range(NUM_ACTIONS)])

Q = onpolicy_monte_carlo(10000, lambda state, Q: random.randrange(NUM_ACTIONS) if random.uniform(0, 1) <= epsilon else get_best_action(Q, state))

print(f"Q (action-value function) (expected returns): {Q}")