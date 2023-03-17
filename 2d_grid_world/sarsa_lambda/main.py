import random
import numpy as np

import matplotlib.pyplot as plt

from sarsa_lambda import sarsa_lambda
from GridWorld import GridWorld, State

env = GridWorld((10, 4), (0, 0), (9, 0), [(x, 0) for x in range(1, 9)], -1, -100, 0)

def policy_to_str(policy,
                  overlay=False,
                  chars=
                  {
                     "cliff": "☠",
                     "state": "😀",
                     "goal": "🏆"
                  }):
   grid = np.split(np.array(list(map(lambda action: str(action), policy))), env.NUM_ROWS)

   if overlay:
      for cliff_tile in env.CLIFF_TILES:
         grid[cliff_tile.y][cliff_tile.x] = chars["cliff"]

      grid[env.GOAL_STATE.y][env.GOAL_STATE.x] = chars["goal"]

      grid[env.STARTING_STATE.y][env.STARTING_STATE.x] = chars["state"]

   return "\n".join(["".join(map(lambda char: f"[{char}]", row)) for row in reversed(grid)])

def print_policy(policy):
   print(f"{policy_to_str(policy)}\n")
   print(f"{policy_to_str(policy, True)}\n")

def get_best_action(Q, state):
   actions = env.get_valid_actions(state)

   return actions[np.argmax([Q[env.state_to_int(state)][action.value] for action in actions])]

def get_optimal_policy(Q):
   return [get_best_action(Q, State(x, y)) for y in range(env.NUM_ROWS) for x in range(env.NUM_COLUMNS)]

epsilon = 0.1

def policy(Q):
   if random.uniform(0, 1) <= epsilon:
      return np.random.choice(env.get_valid_actions())
      
   return get_best_action(Q, env.state)

alpha = 0.5
gamma = 1
episodes = 500

def plot_average_rewards(num_experiments, lambdas):
   plt.figure()
   
   sarsa_lambda_rewards = [[[] for _ in range(num_experiments)] for __ in lambdas]

   for i in range(num_experiments):
      print(f"Experiment {i + 1}/{num_experiments}...")

      for j, tracer_lambda in enumerate(lambdas):
         Q, rewards = sarsa_lambda(env, episodes, alpha, tracer_lambda, gamma, policy)
         sarsa_lambda_rewards[j][i] = rewards

   for rewards in sarsa_lambda_rewards:
      plt.plot([i for i in range(episodes)], np.mean(rewards, axis=0))

   plt.xlabel("Episode")
   plt.ylabel("Averaged sum of rewards")

   plt.legend([f"SARSA(λ = {tracer_lambda})" for tracer_lambda in lambdas])

   plt.show()

plot_average_rewards(100, [0.5, 0.8])

tracer_lambda = 0.6

Q, rewards = sarsa_lambda(env, episodes, tracer_lambda, alpha, gamma, policy)
print(f"SARSA(λ = {tracer_lambda}): Best actions (optimal policy):\n")
print_policy(get_optimal_policy(Q))
