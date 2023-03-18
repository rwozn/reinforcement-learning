import numpy as np

from GridWorld import GridWorld, Action

def sarsa(env: GridWorld, num_episodes: int, alpha: float, gamma: float, policy):
   Q = np.zeros((env.NUM_STATES, Action.num()))
   
   rewards = np.zeros((num_episodes))

   for i in range(num_episodes):
      env.reset()

      action = policy(Q)

      while True:
         state_Q_index = env.state_to_int()

         next_state, reward, done = env.perform_action(action)

         rewards[i] += reward

         next_action = policy(Q)

         prediction = Q[state_Q_index][action.value]

         next_expected_return = Q[env.state_to_int()][next_action.value]

         target = reward + gamma * next_expected_return

         delta = target - prediction

         Q[state_Q_index][action.value] += alpha * delta

         if done:
            break
         
         action = next_action

   return Q, rewards
