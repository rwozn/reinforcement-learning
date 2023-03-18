import random

from rl.Agent import Agent

class RandomAgent(Agent):
   def __init__(self, num_actions):
      super().__init__()
      
      self.num_actions = num_actions
   
   # Sample randomly from the action space
   def choose_action(self, state):
      return random.randrange(self.num_actions)