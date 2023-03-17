import random
import numpy as np

SQUARES = 5
STARTING_POSITION = 0
GOAL_SQUARE = SQUARES - 1

ACTION_LEFT = 0
ACTION_RIGHT = 1

NUM_STATES = GOAL_SQUARE
NUM_ACTIONS = 2

REWARD = SQUARES
LAST_VALID_STATE = GOAL_SQUARE - 1

def is_terminal_state(state):
   return state < STARTING_POSITION or state > LAST_VALID_STATE

def perform_action(state, action):
   state += -1 if action == ACTION_LEFT else 1
   reward = REWARD if state == GOAL_SQUARE else 0

   return state, reward

# SARSA chooses the next action (a') before it updates the
# action-value function. a' updates the policy and directs the agent
# on the next step, which makes this an on-policy algorithm
def sarsa(num_episodes, alpha, gamma, policy):
   Q = [[0] * NUM_ACTIONS for _ in range(NUM_STATES)]

   for _ in range(num_episodes):
      state = STARTING_POSITION

      action = policy(state, Q)

      while True:
         next_state, reward = perform_action(state, action)

         end = is_terminal_state(next_state)

         next_action = -1 if end else policy(next_state, Q)

         next_expected_return = 0 if end else Q[next_state][next_action]

         delta = reward + gamma * next_expected_return - Q[state][action]

         Q[state][action] += alpha * delta

         if end:
            break
         
         state = next_state
         action = next_action

   return Q

def get_best_action(Q, state):
   return np.argmax(Q[state])

epsilon = 0.3

# epsilon-greedy search
# epsilon - exploration probability - im więcej tym częściej eksploruje: 0 <= epsilon <= 1
Q = sarsa(10000, 0.7, 0.9, lambda state, Q: random.randrange(NUM_ACTIONS) if random.uniform(0, 1) <= epsilon else get_best_action(Q, state))

print(f"Q =\n{np.array(Q)}")
print(f"Best actions (optimal policy): {[get_best_action(Q, state) for state in range(NUM_STATES)]}")