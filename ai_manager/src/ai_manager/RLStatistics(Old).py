from RLAlgorithm import RLAlgorithm
import matplotlib.pyplot as plt
import os

rl_algorithm = RLAlgorithm.recover_training(filename="trainings/training20201209.pkl")
image_path = "statistics/"

if not os.path.isdir(image_path):
    os.makedirs(image_path)

print("Steps performed: {}".format(rl_algorithm.statistics.current_step))
print("Episodes performed: {}".format(rl_algorithm.statistics.episode))
print("Amount of pick actions: {}".format(sum(rl_algorithm.statistics.episode_picks)))
successful_episodes = list(filter((False).__ne__, rl_algorithm.statistics.episode_succeed))
print("Percentage of successful episodes: {}".format(100 * len(successful_episodes) / len(rl_algorithm.statistics.episode_succeed)))

# Steps per episodes
try:
       fig, ax = plt.subplots()
       ax.plot(range(1, len(rl_algorithm.statistics.episode_steps) + 1), rl_algorithm.statistics.episode_steps)

       ax.set(xlabel='Episodes', ylabel='Number of steps',
              title='Evolution of number of steps per episode')
       ax.grid()

       fig.savefig(image_path + "steps_per_episode.png")
       plt.show()
except:
       print("Error while plotting steps_per_episode")

# Picks per episodes
try:
       fig, ax = plt.subplots()
       ax.plot(range(1, len(rl_algorithm.statistics.episode_picks) + 1), rl_algorithm.statistics.episode_picks)

       ax.set(xlabel='Episodes', ylabel='Number of picks',
              title='Evolution of number of pick actions per episode')
       ax.grid()

       fig.savefig(image_path + "picks_per_episode.png")
       plt.show()
except:
       print("Error while plotting picks_per_episode")

#  Reward per episodes
try:
       fig, ax = plt.subplots()
       ax.plot(range(1, len(rl_algorithm.statistics.episode_total_reward) + 1), rl_algorithm.statistics.episode_total_reward)

       ax.set(xlabel='Episodes', ylabel='Total reward',
              title='Evolution of total reward per episode')
       ax.grid()

       fig.savefig(image_path + "reward_per_episode.png")
       plt.show()
except:
       print("Error while plotting reward_per_episode")

#  Episode successful
try:
       fig, ax = plt.subplots()
       ax.plot(range(1, len(rl_algorithm.statistics.episode_succeed) + 1), rl_algorithm.statistics.episode_succeed)

       ax.set(xlabel='Episodes', ylabel='Episode successful',
              title='Episode successful')
       ax.grid()

       fig.savefig(image_path + "successful_episode.png")
       plt.show()
except:
       print("Error while plotting successful_episode")

#  Random actions
try:
       fig, ax = plt.subplots()
       ax.plot(range(1, len(rl_algorithm.statistics.episode_random_actions) + 1), rl_algorithm.statistics.episode_random_actions)

       ax.set(xlabel='Episodes', ylabel='Random Actions',
              title='Evolution of Random Actions')
       ax.grid()

       fig.savefig(image_path + "random_actions.png")
       plt.show()
except:
       print("Error while plotting random_actions")