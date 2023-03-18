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

def get_best_action(Q, state):
   return np.argmax(Q[state])

# alpha controls the rate at which you update the action-value estimate, should be: 0 <= alpha <= 1
# It helps to average over noisy updates.
# High values learn faster but perform less averaging.
#
# The algorithm is off-policy because the update only affects the current policy.
# It doesn't use the update to direct the agent. The agent still has to derive the
# action from the current policy.
# This is a subtle, but crucial improvement that has only recently been exploited.
def Q_learning(num_episodes, alpha, gamma, policy):
   Q = [[0] * NUM_ACTIONS for _ in range(NUM_STATES)]

   for _ in range(num_episodes):
      # initialize the environment (reset it)
      state = STARTING_POSITION

      while True:
         action = policy(state, Q)

         next_state, reward = perform_action(state, action)

         end = is_terminal_state(next_state)

         highest_return = 0 if end else Q[next_state][get_best_action(Q, next_state)]

         # delta = a measure of how wrong the estimate is compared to what actually happened.
         # The TD one-step look-ahead implemeneted in the delta allows the algorithm to
         # project one step forward. This allows the agent to ask "Which is the best action?".
         delta = reward + gamma * highest_return - Q[state][action]

         Q[state][action] += alpha * delta

         if end:
            break

         state = next_state

   return Q

epsilon = 0.3

# epsilon-greedy search
# epsilon - exploration probability - the higher epsilon the more often the agent explores: 0 <= epsilon <= 1
Q = Q_learning(10000, 0.7, 0.9, lambda state, Q: random.randrange(NUM_ACTIONS) if random.uniform(0, 1) <= epsilon else get_best_action(Q, state))

print(f"Q =\n{np.array(Q)}")
print(f"Best actions (optimal policy): {[get_best_action(Q, state) for state in range(NUM_STATES)]}")