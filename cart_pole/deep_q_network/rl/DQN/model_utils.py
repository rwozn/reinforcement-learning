import numpy as np
import tensorflow.keras as keras

def create_model(num_states, num_actions, learning_rate=5e-3):
   model = keras.models.Sequential([
                                    keras.layers.Dense(64, input_shape=[num_states], activation="relu"),
                                    keras.layers.Dense(32, activation="relu"),
                                    keras.layers.Dense(num_actions, activation="linear")])

   model.compile(
               loss="mse",
               optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
   
   return model

def get_predicted_action(model, state):
   return np.argmax(model(np.expand_dims(state, axis=0))[0])
   