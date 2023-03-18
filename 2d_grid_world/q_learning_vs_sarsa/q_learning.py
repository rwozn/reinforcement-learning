import numpy as np

from GridWorld import GridWorld, Action

def Q_learning(env: GridWorld, num_episodes, alpha, gamma, policy):
   Q = np.zeros((env.NUM_STATES, Action.num()))

   rewards = np.zeros((num_episodes))

   for i in range(num_episodes):
      env.reset()

      while True:
         action = policy(Q)

         state_Q_index = env.state_to_int()

         next_state, reward, done = env.perform_action(action)

         rewards[i] += reward

         prediction = Q[state_Q_index][action.value]

         highest_return = np.max([Q[env.state_to_int()][action.value] for action in env.get_valid_actions()])

         target = reward + gamma * highest_return

         delta = target - prediction

         Q[state_Q_index][action.value] += alpha * delta

         if done:
            break

   return Q, rewards
