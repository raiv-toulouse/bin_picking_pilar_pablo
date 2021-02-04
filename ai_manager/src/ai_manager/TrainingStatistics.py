import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
# from Environment import Environment
import math


class TrainingStatistics:
    """
       Class were all the statistics of the training will be stored.
       """

    def __init__(self):
        self.current_step = 0  # Current step since the beginning of training
        self.episode = 0  # Number of episode
        self.episode_steps = [0]  # Steps taken by each episode
        self.episode_picks = [0]  # Pick actions tried by each episode
        self.episode_total_reward = [0]  # Total reward of each episode
        self.episode_random_actions = [0]  # Number of random actions in each episode
        self.episode_succeed = []  # Array that stores whether each episode has ended successfully or not
        self.coordinates_matrix = self.generate_coordinates_matrix()

    def generate_coordinates_matrix(self):
        x_limit = 0.13 / 2
        y_limit = 0.19 / 2

        matrix_width = 2 * math.ceil(x_limit / 0.02)
        matrix_height = 2 * math.ceil(y_limit / 0.02)

        return [([0] * matrix_height) for i in range(matrix_width)]

    def fill_coordinates_matrix(self, coordinates):
        try:
            matrix_width = len(self.coordinates_matrix[0])  # y
            matrix_height = len(self.coordinates_matrix)  # x

            x_idx = int(math.ceil(coordinates[0] / 0.02) + (matrix_height / 2) - 1)
            y_idx = int(math.ceil(coordinates[1] / 0.02) + (matrix_width / 2) - 1)

            self.coordinates_matrix[x_idx][y_idx] += 1
        except:
            print('Error while filling coordinates statistics matrix')

    def new_episode(self):
        self.episode += 1  # Increase the episode counter
        self.episode_steps.append(0)  # Append a new value to the next episode step counter
        self.episode_picks.append(0)  # Append a new value to the amount of picks counter
        self.episode_total_reward.append(0)  # Append a new value to the next episode total reward counter
        self.episode_random_actions.append(0)  # Append a new value to the next episode random actions counter

    def new_step(self):
        self.current_step += 1  # Increase step
        self.episode_steps[-1] += 1  # Increase current episode step counter

    def increment_picks(self):
        self.episode_picks[-1] += 1  # Increase of the statistics counter

    def add_reward(self, reward):
        self.episode_total_reward[-1] += reward

    def add_succesful_episode(self, successful):
        self.episode_succeed.append(successful)

    def random_action(self):
        self.episode_random_actions[-1] += 1

    def save(self, filename='trainings/rl_algorithm_stats.pkl'):
        def create_if_not_exist(filename):
            current_path = os.path.dirname(os.path.realpath(__file__))
            filename = os.path.join(current_path, filename)
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            return filename
        filename = create_if_not_exist(filename)
        with open(filename, 'wb+') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def recover(filename='trainings/rl_algorithm_stats.pkl', episode_offset=0, clean_unsuccessful_episodes=False):
        current_path = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(current_path, filename)
        print(filename)
        try:
            with open(filename, 'rb') as input:
                stats = pickle.load(input)

            if clean_unsuccessful_episodes:
                idx = [i + 1 for i in range(len(stats.episode_succeed)) if stats.episode_succeed[i]]

                stats.episode_steps = [stats.episode_steps[i] for i in idx]
                stats.episode_picks = [stats.episode_picks[i] for i in idx]
                stats.episode_total_reward = [stats.episode_total_reward[i] for i in idx]
                stats.episode_random_actions = [stats.episode_random_actions[i] for i in idx]
                stats.episode = len(idx)  # Number of episode

            # Delete de last N episodes
            stats.episode = stats.episode - episode_offset
            stats.episode_steps = stats.episode_steps[:len(stats.episode_steps) - episode_offset]
            stats.episode_picks = stats.episode_picks[:len(stats.episode_picks) - episode_offset]
            stats.episode_total_reward = stats.episode_total_reward[:len(stats.episode_total_reward) - episode_offset]
            stats.episode_random_actions = stats.episode_random_actions[:len(stats.episode_random_actions) - episode_offset]
            stats.episode_succeed = stats.episode_succeed[:len(stats.episode_succeed) - episode_offset]

            # Calculate the percentage of random actions
            stats.episode_random_actions = [
                stats.episode_random_actions[i] * 100 / (stats.episode_steps[i] + 1)
                for i in range(len(stats.episode_steps))]

            stats.current_step = sum(stats.episode_steps)
            return stats
        except IOError:
            print("There is no Training saved in this path.")

    @staticmethod
    def get_average_steps(period, values):
        values = values[-period:]
        return sum(values) / len(values)

    def print_general_info(self):
        print("Steps performed: {}".format(self.current_step))
        print("Episodes performed: {}".format(self.episode))
        print("Amount of pick actions: {}".format(sum(self.episode_picks)))
        successful_episodes = list(filter((False).__ne__, self.episode_succeed))
        print(
            "Percentage of successful episodes: {}".format(100 * len(successful_episodes) / len(self.episode_succeed)))

    def plt_series(self, serie, xlabel='', ylabel='', title='', filename='', plot_type='line', offset=0):
        try:
            if offset > 0:
                serie = [self.get_average_steps(offset, serie[:i+1]) for i in range(len(serie))]
                serie = serie[offset:]
            fig, ax = plt.subplots()
            if plot_type == 'bar':
                ax.bar(range(1, len(serie) + 1), serie)
            else:
                ax.plot(range(1, len(serie) + 1), serie)

            ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
            ax.grid()

            fig.savefig(self.image_path + filename)
            plt.show()
        except:
            print("Error while plotting {}".format(filename))


if __name__ == '__main__':

    stats = TrainingStatistics.recover(

        filename='trainings/bs256_g0.999_es1_ee0.01_ed0.0005_lr_0.0001_optimal_original_rewards_algorithm1901_stats.pkl',
        # filename='trainings/bs256_g0.999_es1_ee0.01_ed0.0005_lr_0.0001_optimal_original_rewards_stats.pkl',
        # filename='trainings/bs256_g0.999_es1_ee0.01_ed0.0005_lr_0.0001_optimal_original_rewards_new_model_stats.pkl',
        episode_offset=0, clean_unsuccessful_episodes=True)

    stats.image_path = "statistics/"

    if not os.path.isdir(stats.image_path):
        os.makedirs(stats.image_path)

    stats.print_general_info()

    offset = 100

    # Steps per episodes
    stats.plt_series(
        serie=stats.episode_steps,
        xlabel='Episodes',
        ylabel='Number of steps',
        title='Evolution of number of steps per episode',
        filename="steps_per_episode.png",
        offset=offset)

    # Picks per episodes
    stats.plt_series(
        serie=stats.episode_picks,
        xlabel='Episodes',
        ylabel='Number of picks',
        title='Evolution of number of pick actions per episode',
        filename="picks_per_episode.png",
        offset=offset)

    #  Reward per episodes
    stats.plt_series(
        serie=stats.episode_total_reward,
        xlabel='Episodes',
        ylabel='Total reward',
        title='Evolution of total reward per episode',
        filename="reward_per_episode.png",
        offset=offset)

    #  Episode successful
    stats.plt_series(
        serie=stats.episode_succeed,
        xlabel='Episodes',
        ylabel='Episode successful',
        title='Successful Episodes',
        filename="successful_episode.png",
        plot_type='bar')

    #  Random actions
    stats.plt_series(
        serie=stats.episode_random_actions,
        xlabel='Episodes',
        ylabel='Random Actions (%)',
        title='Percentage of Random Actions per episode',
        filename="random_actions.png",
        offset=offset
    )

    #  Random actions
    try:
        fig, ax = plt.subplots()
        # Create a dataset from the statistics gader
        df = pd.DataFrame(np.array(stats.coordinates_matrix))

        # Default heatmap: just a visualization of this square matrix
        p1 = sns.heatmap(df, cmap='coolwarm')
        p1.set(xlabel='Y coordinates', ylabel='X coordinates',
                   title='Robot movement heatmap')
        plt.show()
    except:
        print("Error while plotting robot heatmap")
