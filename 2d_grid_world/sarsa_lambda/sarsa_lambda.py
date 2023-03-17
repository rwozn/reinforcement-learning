import numpy as np

from GridWorld import GridWorld, Action

# alpha - step size, 0 < alpha < 1
# lambda - tracer decay rate, 0 <= lambda <= 1
# (named `tracer_lambda` because `lambda` is a keyword
# so it can't be used a variable name)
def sarsa_lambda(env: GridWorld, num_episodes: int, alpha: float, tracer_lambda: float, gamma: float, policy):
   Q_table_shape = (env.NUM_STATES, Action.num())

   Q = np.zeros(Q_table_shape)

   # z is a table that represents the current state-action pair.
   # This is how the algorithm "remembers" which states and actions it
   # touched during the trajectory.
   z = np.zeros(Q_table_shape)

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

         # Implement a tracer by incrementing a cell in the z table.
         z[state_Q_index][action.value] += 1

         # Loop over all states and actions to update the action-value function.
         for state in range(env.NUM_STATES):
            for action in range(Action.num()):
               # Weigh the TD error by the current tracer value, which decays next.
               # In other words, if the agent touched the state-action pair a long
               # time ago there is a negligible update. If it touched it in the
               # last time step, there is a large update.
               Q[state][action] += alpha * delta * z[state][action]

               # The current tracer value exponentially decays.
               z[state][action] *= gamma * tracer_lambda

         if done:
            break
         
         action = next_action

   return Q, rewards
