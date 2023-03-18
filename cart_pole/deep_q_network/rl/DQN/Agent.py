import random
import rl.model_utils as model_utils

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from rl.DQN.ReplayBuffer import ReplayBuffer
from rl.TrainableAgent import TrainableAgent
from rl.DQN.model_utils import create_model, get_predicted_action

GAMMA = 0.99

REPLAY_BATCH_SIZE = 128
REPLAY_BUFFER_LENGTH = 50000

POLICY_MODEL_SAVE_RATE = 200 # episodes
TARGET_MODEL_UPDATE_RATE = 500 # steps
POLICY_MODEL_CHECKPOINT_SAVE_RATE = 100 # episodes

MAX_EPSILON = 1
MIN_EPSILON = 0.05
EPSILON_DECAY_RATE = 0.9976

LEARNING_RATE = 5e-3

POLICY_MODEL_PATH = "policy_model"

"""
The agent that explores the game and should eventually learn how to play the game.
"""
class DQNAgent(TrainableAgent):
   def __init__(self, num_states, num_actions, train=True, load=True):
      super().__init__(train, lambda: create_model(num_states, num_actions, LEARNING_RATE), POLICY_MODEL_PATH if load else None, POLICY_MODEL_SAVE_RATE, POLICY_MODEL_CHECKPOINT_SAVE_RATE)

      self.epsilon = MAX_EPSILON
      
      self.num_actions = num_actions

      self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_LENGTH, REPLAY_BATCH_SIZE)

      if self.training_enabled:
         self.target_model = model_utils.clone_model(self.model)

   def choose_action(self, state):
      epsilon = self.epsilon if self.training_enabled else MIN_EPSILON

      if random.random() < epsilon:
         return random.randrange(self.num_actions)

      return get_predicted_action(self.model, state)
   
   def on_train_episode_step(self, state, action, reward, next_state, done):
      if self.num_steps % TARGET_MODEL_UPDATE_RATE == 0:
         self.update_target_model()
      
      self.replay_buffer.store_experience(state, action, reward, next_state, done)
   
   def update_target_model(self):
      print(f"Updating target model... (num_steps={self.num_steps})")
      
      model_utils.copy_model(self.target_model, self.model)

   def get_train_data(self):
      batch = self.replay_buffer.sample_gameplay_batch()

      # Turns e.g. [(1, 'a'), (2, 'b'), (3, 'c')] into [1, 2, 3], ['a', 'b', 'c']
      batch = list(map(list, zip(*batch)))

      return [np.array(element) for element in batch]

   def on_pre_train(self):
      print(f"epsilon={self.epsilon}")

      self.epsilon = max(self.epsilon * EPSILON_DECAY_RATE, MIN_EPSILON)

   def train(self, train_data):
      states, actions, rewards, next_states, dones = train_data
      
      discounted_Q = np.max(self.target_model(next_states), axis=1)
      
      target_Q = rewards + GAMMA * discounted_Q * (1 - dones)
      
      # https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch/?hl=en
      # Open a GradientTape to record the operations run
      # during the forward pass, which enables auto-differentation.
      with tf.GradientTape() as tape:
         # Run the forward pass of the layer.
         # The operations that the layer applies to its inputs
         # are going to be recorded on the GradientTape.
         #
         # If model(...) was a numpy array then this would be equivalent:
         # model(...)[np.arange(batch_size), actions]
         #
         # It returns Q values for given actions, e.g. if:
         # model(...) = [[5,1], [0.9, 2]], actions = [1, 0]
         # then:
         # predicted_Q = [1, 0.9]
         predicted_Q = tf.gather(self.model(states, training=True), actions, batch_dims=1)

         loss_function = keras.losses.get(self.model.loss)
         
         # Compute the loss value for this minibatch.
         loss = loss_function(target_Q, predicted_Q)

      # Use the gradient tape to automatically retrieve
      # the gradients of the trainable variables with respect to the loss.
      #
      # trainable_variables == trainable_weights (but trainable_variables isn't trainable_weights)
      grads = tape.gradient(loss, self.model.trainable_variables)

      optimizer = self.model.optimizer
      
      # Run one step of gradient descent by updating
      # the value of the variables to minimize the loss.
      optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

      return loss
