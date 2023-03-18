import tensorflow as tf
import rl.model_utils as model_utils

from rl.Agent import Agent
from abc import abstractmethod

class TrainableAgent(Agent):
   def __init__(self, train=True, create_model=None, model_path=None, model_save_rate=100, model_checkpoint_save_rate=100):
      super().__init__()
      
      self.training_enabled = train

      self.model_path = model_path
      self.model_save_rate = model_save_rate
      self.model_checkpoint_save_rate = model_checkpoint_save_rate

      print(f'Training={train}, model: {{path="{model_path}", save_rate={model_save_rate} episodes, checkpoint_save_rate={model_checkpoint_save_rate} episodes}}')

      if train and model_path is None:
         raise ValueError("model_path cannot be None if training is enabled")
      
      # Try to load the model if its pat is given.
      if model_path is not None:
         self.model = model_utils.load_model(model_path)
      
      # If the model isn't to be loaded or there's none at the given path ten create a new one.
      if model_path is None or self.model is None:
         if create_model is None:
            raise ValueError("create_model and model_path cannot be both unspecified")

         self.model = create_model()

      self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0), net=self.model)
      self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, "checkpoints", max_to_keep=5)

      #self.load_checkpoint()
   
   def save_checkpoint(self):
      print("Saving a checkpoint...")

      self.checkpoint_manager.save()
   
   def load_checkpoint(self):
      print("Loading a checkpoint...")

      self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

   def on_episode_step(self, state, action, reward, next_state, done):
      super().on_episode_step(state, action, reward, next_state, done)

      if self.training_enabled:
         self.on_train_episode_step(state, action, reward, next_state, done)
   
   @abstractmethod
   def on_train_episode_step(self, state, action, reward, next_state, done):
      pass
   
   @abstractmethod
   def train(self, train_data):
      pass
   
   @abstractmethod
   def get_train_data(self):
      pass

   @abstractmethod
   def on_pre_train(self):
      pass

   def on_post_train(self):
      if self.num_episodes % self.model_save_rate == 0:
         self.save_model()
      
      if self.num_episodes % self.model_checkpoint_save_rate == 0:
         self.save_checkpoint()
   
   def save_model(self):
      print("Saving model...")

      self.model.save(self.model_path)
   
   """
   Takes a batch of gameplay experiences from replay buffer and
   trains the underlying model with the batch.
   """
   def on_episode_end(self):
      super().on_episode_end()

      if not self.training_enabled:
         return
      
      self.on_pre_train()

      loss = self.train(self.get_train_data())
      
      self.on_post_train()

      return loss
   