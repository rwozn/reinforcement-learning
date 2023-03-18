from abc import ABC, abstractmethod

class Agent(ABC):
   def __init__(self):
      self.num_steps = 0
      self.num_episodes = 0

   def on_episode_end(self):
      self.num_episodes += 1
   
   """
   Takes a state from the game environment and returns an action that should
   be taken given the current state.
   """
   @abstractmethod
   def choose_action(self, state):
      pass

   def preprocess_state(self, state):
      return state
   
   def on_episode_step(self, state, action, reward, next_state, done):
      self.num_steps += 1
