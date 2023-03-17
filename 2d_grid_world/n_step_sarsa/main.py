import random
import numpy as np

import matplotlib.pyplot as plt

from n_step_sarsa import n_step_sarsa
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

def plot_average_rewards(num_experiments, ns):
   plt.figure()
   
   sarsa_rewards = [[[] for _ in range(num_experiments)] for n in ns]

   for i in range(num_experiments):
      print(f"Experiment {i + 1}/{num_experiments}...")

      for j, n in enumerate(ns):
         Q, rewards = n_step_sarsa(n, env, episodes, alpha, gamma, policy)
         sarsa_rewards[j][i] = rewards

   for rewards in sarsa_rewards:
      plt.plot([i for i in range(episodes)], np.mean(rewards, axis=0))

   plt.xlabel("Episode")
   plt.ylabel("Averaged sum of rewards")

   plt.legend([f"\n{n}-Step SARSA" for n in ns])

   plt.show()

plot_average_rewards(100, [1, 2, 4])

n = 4

Q, rewards = n_step_sarsa(n, env, episodes, alpha, gamma, policy)

print(f"{n}-Step SARSA: Best actions (optimal policy):\n")
print_policy(get_optimal_policy(Q))
