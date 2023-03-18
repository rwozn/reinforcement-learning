import random
import numpy as np

import matplotlib.pyplot as plt

from sarsa import sarsa
from q_learning import Q_learning
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

def plot_rewards(rewards, episodes):
   plt.plot([i for i in range(episodes)], rewards)

   plt.xlabel("Episode")
   plt.ylabel("Sum of rewards")

   plt.show()

epsilon = 0.1

def policy(Q):
   if random.uniform(0, 1) <= epsilon:
      return np.random.choice(env.get_valid_actions())
      
   return get_best_action(Q, env.state)

alpha = 0.5
gamma = 1
episodes = 500

def plot_average_rewards(num_experiments):
   plt.figure()
   
   sarsa_rewards = [[] for _ in range(num_experiments)]
   Q_learning_rewards = [[] for _ in range(num_experiments)]

   for i in range(num_experiments):
      print(f"Experiment {i + 1}/{num_experiments}...")

      Q, rewards = Q_learning(env, episodes, alpha, gamma, policy)
      Q_learning_rewards[i] = rewards

      Q, rewards = sarsa(env, episodes, alpha, gamma, policy)
      sarsa_rewards[i] = rewards
   
   # Q-Learning
   plt.plot([i for i in range(episodes)], np.mean(Q_learning_rewards, axis=0))
   
   #SARSA
   plt.plot([i for i in range(episodes)], np.mean(sarsa_rewards, axis=0))

   plt.xlabel("Episode")
   plt.ylabel("Averaged sum of rewards")

   plt.legend(["Q-Learning", "SARSA"])

   plt.show()

plot_average_rewards(100)

Q, rewards = sarsa(env, episodes, alpha, gamma, policy)
print(f"SARSA: Best actions (optimal policy):\n")
print_policy(get_optimal_policy(Q))
plot_rewards(rewards, episodes)

Q, rewards = Q_learning(env, episodes, alpha, gamma, policy)
print(f"Q-Learning: Best actions (optimal policy):\n")
print_policy(get_optimal_policy(Q))
plot_rewards(rewards, episodes)
