import gym

from rl.Agent import Agent

class GameEnvironment:
   def __init__(self, game_name, max_episode_steps=None, render=False):      
      self.env = gym.make(
                        game_name,
                        new_step_api=True,
                        max_episode_steps=max_episode_steps,
                        render_mode="human" if render else None)
      
   def get_num_actions(self):
      return self.env.action_space.n

   def get_num_states(self):
      return len(self.env.observation_space.high)
   
   def run_episode(self, agent: Agent):
      done = False
      total_reward = 0
      state = self.env.reset()

      state = agent.preprocess_state(state)

      while True:
         action = agent.choose_action(state)

         next_state, reward, done, truncated, info = self.env.step(action)
         
         next_state = agent.preprocess_state(next_state)
         
         if truncated:
            done = True

         agent.on_episode_step(state, action, reward, next_state, done)

         total_reward += reward

         if done:
            break

         state = next_state

      return total_reward
      