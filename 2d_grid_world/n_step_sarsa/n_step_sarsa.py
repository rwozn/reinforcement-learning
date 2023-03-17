import numpy as np

from GridWorld import GridWorld, Action

def save_experience(experience_replay_buffer, time_step, n, experience):
   experience_replay_buffer[time_step % (n + 1)] = experience

def get_experience(experience_replay_buffer, time_step, n):
   return experience_replay_buffer[time_step % (n + 1)]

# n - the number of steps in the n-step
# alpha - action-value step size, 0 < alpha < 1
def n_step_sarsa(n, env: GridWorld, num_episodes: int, alpha: float, gamma: float, policy):
   Q = np.zeros((env.NUM_STATES, Action.num()))
   
   rewards = np.zeros((num_episodes))

   for i in range(num_episodes):
      # the step number at the end of the episode
      t = 0
      
      # current step number
      T = np.inf

      # tuple in the form of (state, action, reward)
      # buffer[0] - the oldesy experience, buffer[n-1] = the latest experience
      experience_replay_buffer = [(None, None, None) for _ in range(n + 1)]

      env.reset()

      action = policy(Q)

      experience_replay_buffer[0] = (env.state_to_int(), action.value, None)

      while True:
         # Check if the episode has already ended. If it has, you don't
         # want to take any more actions.
         if t < T:
            next_state, reward, done = env.perform_action(action)

            rewards[i] += reward

            save_experience(experience_replay_buffer, t + 1, n, (env.state_to_int(), None, reward))

            # if that action led to the end of the episode, then set the variable T
            # to denote the step at which the episode came to an end
            if done:
               T = t + 1
            else: # otherwise, choose the next action
               action = policy(Q)

               save_experience(experience_replay_buffer, t + 1, n, (env.state_to_int(), action.value, reward))

         # Tau points to the state-action pair that was n steps ago.
         # Initially, this will be before t=0, so it checks to prevent
         # index-out-of-bounds errors next
         tau = t - n + 1

         # If tau >= 0, then begin to update the action-value functions with the expected return
         if tau >= 0:
            G = sum([gamma ** (i - tau - 1) * get_experience(experience_replay_buffer, i, n)[2] for i in range(tau + 1, min(tau + n, T) + 1)])

            if tau + n < T:
               experience = get_experience(experience_replay_buffer, tau + n, n)
            
               G += gamma ** n * Q[experience[0]][experience[1]]

            experience = get_experience(experience_replay_buffer, tau, n)

            Q[experience[0]][experience[1]] += alpha * (G - Q[experience[0]][experience[1]])
         
         if tau == T - 1:
            break
         
         t += 1

   return Q, rewards