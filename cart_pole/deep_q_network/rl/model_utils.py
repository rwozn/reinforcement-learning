import tensorflow.keras as keras

def load_model(path: str):
   try:
      model = keras.models.load_model(path)

      print(f'Loaded the model from "{path}"')

      return model
   except OSError as error:
      print(f'Unable to load the model from "{path}": {error}')

def copy_model(first: keras.models.Model, other: keras.models.Model):
   first.set_weights(other.get_weights())

def clone_model(other: keras.models.Model, copy_weights=True):
   print("Cloning the model...")

   model = keras.models.clone_model(other)

   if copy_weights:
      print("(Also copying weights...)")
      
      copy_model(model, other)
   
   return model
