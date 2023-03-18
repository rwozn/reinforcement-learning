import random
import numpy as np

from collections import deque

class ReplayBuffer:
   def __init__(self, buffer_size, batch_size):
      self.batch_size = batch_size
      self.experiences = deque(maxlen=buffer_size)
   
   """
   Stores a step of gameplay experience in the buffer for later training.
   """
   def store_experience(self, state, action, reward, next_state, done):
      self.experiences.append((state, action, reward, next_state, done))
   
   """
   Samples a batch of gameplay experiences for training purposes.
   """
   def sample_gameplay_batch(self):
      batch_size = min(self.batch_size, len(self.experiences))

      return random.sample(self.experiences, batch_size)