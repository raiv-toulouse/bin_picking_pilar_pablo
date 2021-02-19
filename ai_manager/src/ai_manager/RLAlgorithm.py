# coding=utf-8
import math
import random
import os
import errno
import sys
from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt
import rospy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image

from Environment import Environment
from TrainingStatistics import TrainingStatistics
from ImageProcessing.ImageModel import ImageModel
from ImageController import ImageController

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

import pickle

State = namedtuple(  # State information namedtuple
    'State',
    ('coordinate_x', 'coordinate_y', 'pick_probability', 'object_gripped', 'image_raw')
)

Experience = namedtuple(  # Replay Memory Experience namedtuple
    'Experience',
    ('state', 'coordinates', 'pick_probability', 'action', 'next_state', 'next_coordinates', 'next_pick_probability',
     'reward', 'is_final_state')
)


class RLAlgorithm:
    """
    Class used to perform actions related to the RL Algorithm training. It can be initialized with custom parameters or
    with the default ones.

    To perform a Deep Reinforcement Learning training, the following steps have to be followed:

        1. Initialize replay memory capacity.
        2. Initialize the policy network with random weights.
        3. Clone the policy network, and call it the target network.
        4. For each episode:
           1. Initialize the starting state.
            2. For each time step:
                1. Select an action.
                    - Via exploration or exploitation
                2. Execute selected action in an emulator or in Real-life.
                3. Observe reward and next state.
                4. Store experience in replay memory.
                5. Sample random batch from replay memory.
                6. Preprocess states from batch.
                7. Pass batch of preprocessed states to policy network.
                8. Calculate loss between output Q-values and target Q-values.
                    - Requires a pass to the target network for the next state
                9. Gradient descent updates weights in the policy network to minimize loss.
                    - After time steps, weights in the target network are updated to the weights in the policy network.

    """

    def __init__(self, object_gripped_reward=10, object_not_picked_reward=-10, out_of_limits_reward=-10,
                 horizontal_movement_reward=-1, batch_size=32, gamma=0.999, eps_start=1, eps_end=0.01, eps_decay=0.0005,
                 target_update=10, memory_size=100000, lr=0.001, num_episodes=1000, include_pick_prediction=False,
                 save_training_others='optimal'):
        """

        :param object_gripped_reward: Object gripped reward
        :param object_not_picked_reward: Object not picked reward
        :param out_of_limits_reward: Out of limits reward
        :param horizontal_movement_reward: Horizontal movement reward
        :param batch_size: Size of the batch used to train the network in every step
        :param gamma: discount factor used in the Bellman equation
        :param eps_start: Greedy strategy epsilon start (Probability of random choice)
        :param eps_end: Greedy strategy minimum epsilon (Probability of random choice)
        :param eps_decay: Greedy strategy epsilon decay (Probability decay of random choice)
        :param target_update: How frequently, in terms of episodes, target network will update the weights with the
        policy network weights
        :param memory_size: Capacity of the replay memory
        :param lr: Learning rate of the Deep Learning algorithm
        :param num_episodes:  Number of episodes on training
        :param include_pick_prediction: Use the image model pick prediction as input of the DQN
        :param self_training_others: Parameter used to modify the filename of the training while saving
        """

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.memory_size = memory_size
        self.lr = lr
        self.num_episodes = num_episodes
        self.include_pick_prediction = include_pick_prediction
        self.save_training_others = save_training_others

        self.current_state = None  # Robot current state
        self.previous_state = None  # Robot previous state
        self.current_action = None  # Robot current action
        self.current_action_idx = None  # Robot current action Index
        self.episode_done = False  # True if the episode has just ended

        # This tells PyTorch to use a GPU if its available, otherwise use the CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Torch devide
        self.em = self.EnvManager(self, object_gripped_reward, object_not_picked_reward, out_of_limits_reward,
                                  horizontal_movement_reward)  # Robot Environment Manager
        self.strategy = self.EpsilonGreedyStrategy(self.eps_start, self.eps_end, self.eps_decay)  # Greede Strategy
        self.agent = self.Agent(self)  # RL Agent
        self.memory = self.ReplayMemory(self.memory_size)  # Replay Memory
        self.statistics = TrainingStatistics()  # Training statistics

        self.policy_net = self.DQN(self.em.image_tensor_size,
                                   self.em.num_actions_available(),
                                   self.include_pick_prediction).to(self.device)  # Policy Q Network
        self.target_net = self.DQN(self.em.image_tensor_size,
                                   self.em.num_actions_available(),
                                   self.include_pick_prediction).to(self.device)  # Target Q Network
        self.target_net.load_state_dict(self.policy_net.state_dict())  # Target net has to be the same as policy network
        self.target_net.eval()  # Target net has to be the same as policy network
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)  # Q Networks optimizer

        print("Device: ", self.device)

    class Agent:
        """
        Class that contains all needed methods to control the agent through the environment and retrieve information of
        Its state
        """

        def __init__(self, rl_algorithm):
            """

            :param self: RLAlgorithm object
            """
            self.strategy = rl_algorithm.strategy  # Greedy Strategy
            self.num_actions = rl_algorithm.em.num_actions_available()  # Num of actions available
            self.device = rl_algorithm.device  # Torch device
            self.rl_algorithm = rl_algorithm

        def select_action(self, state, policy_net):
            """
            Method used to pick the following action of the robot
            Method used to pick the following action of the robot
            :param state: State RLAlgorithm namedtuple with all the information of the current state
            :param policy_net: DQN object used as policy network for the RL algorithm
            :return:
            """
            random_action = False
            if self.rl_algorithm.episode_done:  # If the episode has just ended we reset the robot environment
                self.rl_algorithm.episode_done = False  # Put the variable episode_done back to False
                self.rl_algorithm.statistics.new_episode()

                self.rl_algorithm.current_action = 'random_state'  # Return random_state to reset the robot position
                self.rl_algorithm.current_action_idx = None
            else:
                rate = self.strategy.get_exploration_rate(
                    self.rl_algorithm.statistics.current_step)  # We get the current epsilon value

                if rate > random.random():  # With a probability = rate we choose a random action (Explore environment)
                    action = random.randrange(self.num_actions)
                    random_action = True
                else:  # With a probability = (1 - rate) we Explote the information we already have
                    try:
                        with torch.no_grad():  # We calculate the action using the Policy Q Network
                            action = policy_net(state.image_raw, torch.tensor(
                                [[state.coordinate_x, state.coordinate_y]], device=self.device),
                                                state.pick_probability).argmax(dim=1).to(
                                self.device)  # exploit
                    except:
                        print("Ha habido un error")

                self.rl_algorithm.current_action = self.rl_algorithm.em.actions[action]
                self.rl_algorithm.current_action_idx = action

            return self.rl_algorithm.current_action, random_action  # We return the action as a string, not as int

    class DQN(nn.Module):
        """
        Class to create a Deep Q Learning Neural Network
        """

        def __init__(self, image_tensor_size, num_actions, include_pick_prediction):
            """

            :param image_tensor_size: Size of the input tensor
            :param num_actions: Number of actions, which is the output of the Neural Network
            """
            super(RLAlgorithm.DQN, self).__init__()

            self.linear1 = nn.Linear(image_tensor_size, int(image_tensor_size / 2))
            self.linear2 = nn.Linear(int(image_tensor_size / 2), int(image_tensor_size / 4))
            extra_features = 2  # coordinates
            if include_pick_prediction:
                extra_features = 3  # pick prediction
            self.linear3 = nn.Linear(int(image_tensor_size / 4) + extra_features, num_actions)
            self.linear = nn.Linear(image_tensor_size + 2, num_actions)

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, image_raw, coordinates, pick_probability):

            output = self.linear1(image_raw)
            output = self.linear2(output)
            if pick_probability:
                output = torch.cat((output, coordinates, pick_probability), 1)
            else:
                output = torch.cat((output, coordinates), 1)
            return self.linear3(output)

    class EnvManager:
        """
        Class used to manage the RL environment. It is used to perform actions such as calculate rewards or retrieve the
        current state of the robot.
        """

        def __init__(self, rl_algorithm, object_gripped_reward, object_not_picked_reward, out_of_limits_reward,
                     horizontal_movement_reward):
            """
            Initialization of an object
            :param rl_manager: RLAlgorithm object
            :param object_gripped_reward: Object gripped reward
            :param object_not_picked_reward: Object not picked reward
            :param out_of_limits_reward: Out of limits reward
            :param horizontal_movement_reward: Horizontal movement reward
            """
            self.object_gripped_reward = object_gripped_reward
            self.out_of_limits_reward = out_of_limits_reward
            self.object_not_picked_reward = object_not_picked_reward
            self.horizontal_movement_reward = horizontal_movement_reward

            self.device = rl_algorithm.device  # Torch device
            self.image_controller = ImageController()  # ImageController object to manage images
            self.actions = ['north', 'south', 'east', 'west', 'pick']  # Possible actions of the objects
            self.image_height = None  # Retrieved images height
            self.image_width = None  # Retrieved Images Width
            self.image = None  # Current image ROS message
            self.image_tensor = None  # Current image tensor
            self.pick_probability = None  # Current image tensor

            #self.model_name = 'model-epoch=05-val_loss=0.36-weights7y3_unfreeze2.ckpt'
            self.model_name = 'resnet50_freezed.ckpt'
            self.model_family = 'resnet50'
            self.image_model = ImageModel(model_name=self.model_family)
            self.feature_extraction_model = self.image_model.load_model(self.model_name)
            self.image_tensor_size = self.image_model.get_size_features(
                self.feature_extraction_model)  # Size of the image after performing some transformations

            self.rl_algorithm = rl_algorithm
            self.gather_image_state()  # Retrieve initial state image

        def calculate_reward(self, previous_image):
            """
            Method used to calculate the reward of the previous action and whether it is a final state or not
            :return: reward, is_final_state
            """
            current_coordinates = [self.rl_algorithm.current_state.coordinate_x,
                                   self.rl_algorithm.current_state.coordinate_y]  # Retrieve robot's current coordinates
            object_gripped = self.rl_algorithm.current_state.object_gripped  # Retrieve if the robot has an object gripped
            if Environment.is_terminal_state(current_coordinates, object_gripped):  # If is a terminal state
                self.rl_algorithm.episode_done = True  # Set the episode_done variable to True to end up the episode
                episode_done = True
                if object_gripped:  # If object_gripped is True, the episode has ended successfully
                    reward = self.object_gripped_reward
                    self.rl_algorithm.statistics.add_succesful_episode(True)  # Saving episode successful statistic
                    self.rl_algorithm.statistics.increment_picks()  # Increase of the statistics cpunter
                    rospy.loginfo("Episode ended: Object gripped!")
                    self.image_controller.record_image(previous_image, True)  # Saving the falure state image
                else:  # Otherwise the robot has reached the limits of the environment
                    reward = self.out_of_limits_reward
                    self.rl_algorithm.statistics.add_succesful_episode(False)  # Saving episode failure statistic
                    rospy.loginfo("Episode ended: Environment limits reached!")
            else:  # If it is not a Terminal State
                episode_done = False
                if self.rl_algorithm.current_action == 'pick':  # if it is not the first action and action is pick
                    reward = self.object_not_picked_reward
                    self.image_controller.record_image(previous_image, False)  # Saving the falure state image
                    self.rl_algorithm.statistics.increment_picks()  # Increase of the statistics counter
                else:  # otherwise
                    self.rl_algorithm.statistics.fill_coordinates_matrix(current_coordinates)
                    reward = self.horizontal_movement_reward

            self.rl_algorithm.statistics.add_reward(reward)  # Add reward to the algorithm statistics
            return reward, episode_done

        def gather_image_state(self):
            """
            This method gather the relative state of the robot by retrieving an image using the image_controller class,
            which reads the image from the ROS topic specified.
            """
            previous_image = self.image
            self.image, self.image_width, self.image_height = self.image_controller.get_image()  # We retrieve state image
            self.image_tensor, pick_probability = self.extract_image_features(self.image)
            if self.rl_algorithm.include_pick_prediction:
                self.pick_probability = pick_probability
            return previous_image

        def extract_image_features(self, image):
            """
            Method used to transform the image to extract image features by passing it through the image_model CNN
            network
            :param image_raw: Image
            :return:
            """
            features, pick_prediction = self.image_model.evaluate_image(image, self.feature_extraction_model)
            features = torch.from_numpy(features)
            return features.to(self.device), torch.tensor([math.exp(pick_prediction.numpy()[0][1])]).to(self.device)

        def num_actions_available(self):
            """
            Returns the number of actions available
            :return: Number of actions available
            """
            return len(self.actions)

    class EpsilonGreedyStrategy:
        """
        Class used to perform the Epsilon greede strategy
        """

        def __init__(self, start, end, decay):
            """
            Initialization
            :param start: Greedy strategy epsilon start (Probability of random choice)
            :param end: Greedy strategy minimum epsilon (Probability of random choice)
            :param decay: Greedy strategy epsilon decay (Probability decay of random choice)
            """
            self.start = start
            self.end = end
            self.decay = decay

        def get_exploration_rate(self, current_step):
            """
            It calculates the rate depending on the actual step of the execution
            :param current_step: step of the training
            :return:
            """
            return self.end + (self.start - self.end) * \
                   math.exp(-1. * current_step * self.decay)

    class QValues:
        """
        It returns the predicted q-values from the policy_net for the specific state-action pairs that were passed in.
        states and actions are the state-action pairs that were sampled from replay memory.
        """

        @staticmethod
        def get_current(policy_net, states, coordinates, actions, pick_probabilities):
            """
            With the current state of the policy network, it calculates the q_values of
            :param policy_net: policy network used to decide the actions
            :param states: Set of state images (Preprocessed)
            :param coordinates: Set of robot coordinates
            :param actions: Set of taken actions
            :return:
            """
            return policy_net(states, coordinates, pick_probabilities).gather(dim=1, index=actions.unsqueeze(-1))

        @staticmethod
        def get_next(target_net, next_states, next_coordinates, next_pick_probabilities, is_final_state):
            """
            Calculate the maximum q-value predicted by the target_net among all possible next actions.
            If the action has led to a terminal state, next reward will be 0. If not, it is calculated using the target
            net
            :param target_net: Target Deep Q Network
            :param next_states: Next states images
            :param next_coordinates: Next states coordinates
            :param is_final_state: Tensor indicating whether this action has led to a final state or not.
            :return:
            """
            batch_size = next_states.shape[0]  # The batch size is taken from next_states shape
            # q_values is initialized with a zeros tensor of batch_size and if there is GPU it is loaded to it
            q_values = torch.zeros(batch_size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            non_final_state_locations = (is_final_state == False)  # Non final state index locations are calculated
            non_final_states = next_states[non_final_state_locations]  # non final state images
            non_final_coordinates = next_coordinates[non_final_state_locations]  # non final coordinates
            non_final_pick_probabilities = next_pick_probabilities[
                non_final_state_locations]  # non final pick probabilities
            # Max q values of the non final states are calculated using the target net
            q_values[non_final_state_locations] = \
                target_net(non_final_states, non_final_coordinates, non_final_pick_probabilities).max(dim=1)[
                    0].detach()
            return q_values

    class ReplayMemory:
        """
        Class used to create a Replay Memory for the RL algorithm
        """

        def __init__(self, capacity):
            """
            Initialization of ReplayMemory
            :param capacity: Capacity of Replay Memory
            """
            self.capacity = capacity
            self.memory = []  # Actual memory. it will be filled with Experience namedtuples
            self.push_count = 0  # will be used to keep track of how many experiences have been added to the memory

        def push(self, experience):
            """
            Method used to fill the Replay Memory with experiences
            :param experience: Experience namedtuple
            :return:
            """
            if len(self.memory) < self.capacity:  # if memory is not full, new experience is appended
                self.memory.append(experience)
            else:  # If its full, we add a new experience and take the oldest out
                self.memory[self.push_count % self.capacity] = experience
            self.push_count += 1  # we increase the memory counter

        def sample(self, batch_size):
            """
            Returns a random sample of experiences
            :param batch_size: Number of randomly sampled experiences returned
            :return: random sample of experiences (Experience namedtuples)
            """
            return random.sample(self.memory, batch_size)

        def can_provide_sample(self, batch_size):
            """
            returns a boolean telling whether or not we can sample from memory. Recall that the size of a sample
            we’ll obtain from memory will be equal to the batch size we use to train our network.
            :param batch_size: Batch size to train the network
            :return: boolean telling whether or not we can sample from memory
            """
            return len(self.memory) >= batch_size

    def extract_tensors(self, experiences):
        """
        Converts a batch of Experiences to Experience of batches and returns all the elements separately.
        :param experiences: Batch of Experienc objects
        :return: A tuple of each element of a Experience namedtuple
        """
        batch = Experience(*zip(*experiences))

        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        coordinates = torch.cat(batch.coordinates)
        next_coordinates = torch.cat(batch.next_coordinates)
        pick_probabilities = torch.cat(batch.pick_probability)
        next_pick_probabilities = torch.cat(batch.next_pick_probability)
        is_final_state = torch.cat(batch.is_final_state)

        return states, coordinates, pick_probabilities, actions, rewards, next_states, next_coordinates, \
               next_pick_probabilities, is_final_state

    @staticmethod
    def saving_name(batch_size, gamma, eps_start, eps_end, eps_decay, lr, others=''):
        return 'bs{}_g{}_es{}_ee{}_ed{}_lr_{}_{}.pkl'.format(
            batch_size, gamma, eps_start, eps_end, eps_decay, lr, others
        )

    def save_training(self, dir='trainings/', others='optimal'):

        filename = self.saving_name(self.batch_size, self.gamma, self.eps_start, self.eps_end, self.eps_decay, self.lr,
                                    self.save_training_others)

        def create_if_not_exist(filename, dir):
            current_path = os.path.dirname(os.path.realpath(__file__))
            filename = os.path.join(current_path, dir, filename)
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            return filename

        rospy.loginfo("Saving training...")

        abs_filename = create_if_not_exist(filename, dir)

        self.em.image_model = None
        self.em.feature_extraction_model = None

        with open(abs_filename, 'wb+') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

        rospy.loginfo("Saving Statistics...")

        filename = 'trainings/{}_stats.pkl'.format(filename.split('.pkl')[0])
        self.statistics.save(filename=filename)

        rospy.loginfo("Training saved!")

    @staticmethod
    def recover_training(batch_size=32, gamma=0.999, eps_start=1, eps_end=0.01,
                         eps_decay=0.0005, lr=0.001, others='optimal', dir='trainings/', ):
        current_path = os.path.dirname(os.path.realpath(__file__))
        filename = RLAlgorithm.saving_name(batch_size, gamma, eps_start, eps_end, eps_decay, lr, others)
        filename = os.path.join(current_path, dir, filename)
        try:
            with open(filename, 'rb') as input:
                rl_algorithm = pickle.load(input)
                rospy.loginfo("Training recovered. Next step will be step number {}"
                              .format(rl_algorithm.statistics.current_step))

                rl_algorithm.em.image_model = ImageModel(model_name=rl_algorithm.em.model_family)
                rl_algorithm.em.feature_extraction_model = rl_algorithm.em.image_model.load_model(
                    rl_algorithm.em.model_name)

                return rl_algorithm
        except IOError:
            rospy.loginfo("There is no Training saved. New object has been created")
            return RLAlgorithm(batch_size=batch_size, gamma=gamma, eps_start=eps_start, eps_end=eps_end,
                               eps_decay=eps_decay, lr=lr, include_pick_prediction=True, save_training_others=others)

    def train_net(self):
        """
        Method used to train both the train and target Deep Q Networks. We train the network minimizing the loss between
        the current Q-values of the action-state tuples and the target Q-values. Target Q-values are calculated using
        thew Bellman's equation:

        q*(state, action) = Reward + gamma * max( q*(next_state, next_action) )
        :return:
        """
        # If there are at least as much experiences stored as the batch size
        if self.memory.can_provide_sample(self.batch_size):
            experiences = self.memory.sample(self.batch_size)  # Retrieve the experiences
            # We split the batch of experience into different tensors
            states, coordinates, pick_probabilities, actions, rewards, next_states, next_coordinates, \
            next_pick_probabilities, is_final_state = self.extract_tensors(experiences)
            # To compute the loss, current_q_values and target_q_values have to be calculated
            current_q_values = self.QValues.get_current(self.policy_net, states, coordinates, actions,
                                                        pick_probabilities)
            # next_q_values is the maximum Q-value of each future state
            next_q_values = self.QValues.get_next(self.target_net, next_states, next_coordinates,
                                                  next_pick_probabilities, is_final_state)
            target_q_values = (next_q_values * self.gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))  # Loss is calculated
            self.optimizer.zero_grad()  # set all the gradients to 0 (initialization) so that we don't accumulate
            # gradient throughout all the backpropagation
            loss.backward(
                retain_graph=True)  # Compute the gradient of the loss with respect to all the weights and biases in the
            # policy net
            self.optimizer.step()  # Updates the weights and biases with the gradients computed

        if self.statistics.episode % self.target_update == 0:  # If target_net has to be updated in this episode
            self.target_net.load_state_dict(self.policy_net.state_dict())  # Target net is updated

    def next_training_step(self, current_coordinates, object_gripped):
        """
        This method implements the Reinforcement Learning algorithm to control the UR3 robot.  As the algorithm is prepared
        to be executed in real life, rewards and final states cannot be received until the action is finished, which is the
        beginning of next loop. Therefore, during an execution of this function, an action will be calculated and the
        previous action, its reward and its final state will be stored in the replay memory.
        :param current_coordinates: Tuple of float indicating current coordinates of the robot
        :param object_gripped: Boolean indicating whether or not ann object has been gripped
        :return: action taken
        """
        self.statistics.new_step()  # Add new steps statistics
        self.previous_state = self.current_state  # Previous state information to store in the Replay Memory
        previous_action = self.current_action  # Previous action to store in the Replay Memory
        previous_action_idx = self.current_action_idx  # Previous action index to store in the Replay Memory
        previous_image = self.em.gather_image_state()  # Gathers current state image

        self.current_state = State(current_coordinates[0], current_coordinates[1], self.em.pick_probability,
                                   object_gripped, self.em.image_tensor)  # Updates current_state

        # Calculates previous action reward an establish whether the current state is terminal or not
        previous_reward, is_final_state = self.em.calculate_reward(previous_image)
        action, random_action = self.agent.select_action(self.current_state,
                                                         self.policy_net)  # Calculates action

        # There are some defined rules that the next action have to accomplish depending on the previous action
        action_ok = False
        while not action_ok:
            # Its forbidden to perform two cosecutive pick actions in the same place
            if action == 'pick' and previous_action != 'pick':
                action_ok = True
            # If previous action was south, it is forbidden to perform a 'north' action for
            # The robot not to go back to the original position.
            elif action == 'north' and previous_action != 'south':
                action_ok = True
            # If previous action was north, it is forbidden to perform a 'south' action for
            # The robot not to go back to the original position.
            elif action == 'south' and previous_action != 'north':
                action_ok = True
            # If previous action was east, it is forbidden to perform a 'west' action for
            # The robot not to go back to the original position.
            elif action == 'west' and previous_action != 'east':
                action_ok = True
            # If previous action was west, it is forbidden to perform a 'east' action for
            # The robot not to go back to the original position.
            elif action == 'east' and previous_action != 'west':
                action_ok = True
            elif action == 'random_state':
                action_ok = True
            else:
                action, random_action = self.agent.select_action(self.current_state,
                                                                 self.policy_net)  # Calculates action

        if random_action:
            self.statistics.random_action()  # Recolecting statistics

        # Random_state actions are used just to initialize the environment to a random position, so it is not taken into
        # account while storing state information in the Replay Memory.
        # If previous action was a random_state and it is not the first step of the training
        if previous_action != 'random_state' and self.statistics.current_step > 1:
            self.memory.push(  # Pushing experience to Replay Memory
                Experience(  # Using an Experience namedtuple
                    self.previous_state.image_raw,  # Initial state image
                    torch.tensor([[self.previous_state.coordinate_x, self.previous_state.coordinate_y]],
                                 device=self.device),  # Initial coordinates
                    self.previous_state.pick_probability,
                    torch.tensor([previous_action_idx], device=self.device),  # Action taken
                    self.current_state.image_raw,  # Final state image
                    torch.tensor([[self.current_state.coordinate_x,
                                   self.current_state.coordinate_y]],
                                 device=self.device),  # Final coordinates
                    self.current_state.pick_probability,
                    torch.tensor([previous_reward], device=self.device),  # Action reward
                    torch.tensor([is_final_state], device=self.device)  # Episode ended
                ))

            # Logging information
            rospy.loginfo("Step: {}, Episode: {}, Previous reward: {}, Previous action: {}".format(
                self.statistics.current_step - 1,
                self.statistics.episode,
                previous_reward,
                previous_action))

            self.train_net()  # Both policy and target networks gets trained

        return action
