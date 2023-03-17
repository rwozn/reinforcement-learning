import numpy as np

from enum import Enum

class Action(Enum):
   LEFT = 0
   RIGHT = 1
   UP = 2
   DOWN = 3

   @staticmethod
   def num():
      return len(list(Action))
   
   def __str__(self):
      if self == Action.LEFT:
         return "←"
      
      if self == Action.RIGHT:
         return "→"

      if self == Action.UP:
         return "↑"

      if self == Action.DOWN:
         return "↓"

      raise ValueError(f"Invalid action")

class State:
   def __init__(self, array):
      self.x, self.y = array

   def __init__(self, x, y):
      self.x = x
      self.y = y

   def perform_action(self, action):
      if action == Action.LEFT:
         self.x -= 1
      elif action == Action.RIGHT:
         self.x += 1
      elif action == Action.UP:
         self.y += 1
      elif action == Action.DOWN:
         self.y -= 1
      else:
         raise ValueError(f"Invalid action: {action}")
      
      return self
   
   def in_bounds(self, num_rows, num_columns):
      return self.x >= 0 and self.x <= num_columns and self.y >= 0 and self.y <= num_rows
   
   @classmethod
   def copy(cls, other):
      return cls(other.x, other.y)
   
   @classmethod
   def from_array(cls, array):
      return cls(array[0], array[1])
   
   def __eq__(self, other) -> bool:
      return self.x == other.x and self.y == other.y

   def __str__(self):
      return f"{{x: {self.x}, y: {self.y}}}"
   
class GridWorld:
   def __init__(self, world_shape, starting_position, goal_position, cliff_tiles, step_reward, die_reward, goal_reward):
      self.NUM_COLUMNS, self.NUM_ROWS = world_shape

      self.NUM_STATES = self.NUM_ROWS * self.NUM_COLUMNS
      self.STARTING_STATE = State.from_array(starting_position)
      self.GOAL_STATE = State.from_array(goal_position)
      self.CLIFF_TILES = [State.from_array(array) for array in cliff_tiles]
      self.STEP_REWARD = step_reward
      self.DIE_REWARD = die_reward
      self.GOAL_REWARD = goal_reward

      self.reset()

   def reset(self):
      self.state = State.copy(self.STARTING_STATE)
   
   def is_dead(self):
      for cliff_tile in self.CLIFF_TILES:
         if self.state == cliff_tile:
            return True
      
      return False
   
   def is_in_goal_state(self):
      return self.state == self.GOAL_STATE
   
   def get_reward(self):
      if self.is_in_goal_state():
         return self.GOAL_REWARD
      
      if self.is_dead():
         return self.DIE_REWARD
      
      return self.STEP_REWARD

   def is_terminal_state(self):
      return self.is_in_goal_state() or self.is_dead()
   
   def get_valid_actions(self, state=None):
      actions = []

      if state is None:
         state = self.state
      
      for action in list(Action):
         if State.copy(state).perform_action(action).in_bounds(self.NUM_ROWS - 1, self.NUM_COLUMNS - 1):
            actions.append(action)
      
      return actions
   
   def is_action_valid(self, action):
      return action in self.get_valid_actions()

   def perform_action(self, action):
      next_state = self.state.perform_action(action)

      return next_state, self.get_reward(), self.is_terminal_state()
   
   def state_to_int(self, state=None):
      if state is None:
         state = self.state

      return state.y * self.NUM_COLUMNS + state.x

   def __str__(self, chars=
                           {
                              "cliff": "☠",
                              "state": "😀",
                              "goal": "🏆",
                              "empty": "#"
                           }):
      grid = [[chars["empty"]] * self.NUM_ROWS for _ in range(self.NUM_COLUMNS)]

      for cliff_tile in self.CLIFF_TILES:
         grid[cliff_tile.x][cliff_tile.y] = chars["cliff"]

      grid[self.GOAL_STATE.x][self.GOAL_STATE.y] = chars["goal"]

      grid[self.state.x][self.state.y] = chars["state"]

      for action in self.get_valid_actions():
         state = State.copy(self.state).perform_action(action)

         grid[state.x][state.y] = str(action)

      return "\n".join(["".join(map(lambda char: f"[{char}]", row)) for row in reversed(np.column_stack(grid))])
