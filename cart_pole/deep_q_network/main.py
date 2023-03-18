import numpy as np
import matplotlib.pyplot as plt

from rl.DQN.Agent import DQNAgent
from rl.RandomAgent import RandomAgent
from rl.GameEnvironment import GameEnvironment

TEST_DURING_TRAINING_RATE = 1000 # episodes
TEST_DURING_TRAINING_DURATION = 15 # episodes

TRAINING_DURATION = 2000 # episodes
TEST_DURATION = 100 # episodes

def test_model(num_episodes, dqn_agent=None, plot=True):
   game_env = GameEnvironment("CartPole-v1", max_episode_steps=200, render=True)

   agent_provided = dqn_agent is not None

   if agent_provided:
      training_enabled = dqn_agent.training_enabled

      dqn_agent.training_enabled = False
   else:
      dqn_agent = DQNAgent(game_env.get_num_states(), game_env.get_num_actions(), False)

   rewards = [0] * num_episodes

   for i in range(num_episodes):
      rewards[i] = game_env.run_episode(dqn_agent)

      print(f"Episode {i + 1}/{num_episodes}: reward={rewards[i]}")

   if plot:
      plot_metrics("Reward", [rewards], ["DQN"])

   # set training_enabled back if training was enabled
   if agent_provided and training_enabled:
      dqn_agent.training_enabled = training_enabled

def train_model(num_episodes):
   game_env = GameEnvironment("CartPole-v1", max_episode_steps=200)

   dqn_agent = DQNAgent(game_env.get_num_states(), game_env.get_num_actions())
   dqn_losses = [0] * num_episodes
   dqn_rewards = [0] * num_episodes

   for i in range(num_episodes):
      dqn_rewards[i] = game_env.run_episode(dqn_agent)

      dqn_losses[i] = dqn_agent.on_episode_end()

      num_episode = i + 1

      print(f"Episode {num_episode}/{num_episodes}: reward={dqn_rewards[i]}, loss={dqn_losses[i]}")

      if num_episode % TEST_DURING_TRAINING_RATE == 0:
         print(f"Testing during training ({int(num_episode / TEST_DURING_TRAINING_RATE)}/{int(num_episodes / TEST_DURING_TRAINING_RATE)})")

         test_model(TEST_DURING_TRAINING_DURATION, dqn_agent, False)

   random_agent = RandomAgent(game_env.get_num_actions())
   random_rewards = [game_env.run_episode(random_agent) for i in range(num_episodes)]

   plot_metrics("Loss", [dqn_losses], ["DQN"])
   plot_metrics("Reward", [dqn_rewards, random_rewards], ["DQN", "Random"])

def plot_metrics(metric_name, metrics, legend):
   plt.figure()
   
   for metric in metrics:
      plt.plot(np.arange(len(metric)), metric)

   plt.xlabel("Episode")
   plt.ylabel(metric_name)

   plt.legend(legend)

   plt.show()

train_model(TRAINING_DURATION)
test_model(TEST_DURATION)