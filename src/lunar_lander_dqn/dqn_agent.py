"""
This module provides an implementation of the Deep Q-Network (DQN) algorithm for training an
agent on the lunar lander gym environment.

Classes:
- AgentDQN: An implementation of the DQN agent.

"""
from collections import deque
from typing import Dict, Union
import numpy as np
import gym
import torch
from ray import tune
from numpy import float64, int64, ndarray
from torch.utils.tensorboard import SummaryWriter
from lunar_lander_dqn.nn_utils import train
from lunar_lander_dqn.nn_utils import predict

CURRENT_STATE_INDEX = 0
ACTION_INDEX = 1
REWARD_INDEX = 2
NEXT_STATE_INDEX = 3
DONE_INDEX = 4

GAMMA = 0.99
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def report_tune(avg_reward: float64) -> None:
    """
    Reports the average reward obtained by the agent in a Ray Tune session.

    Parameters:
    -----------
    avg_reward: float
        The average reward obtained by the agent in an episode.
    """
    if tune.is_session_enabled():
        tune.report(reward=avg_reward)


class AgentDQN:
    """
    The AgentDQN class implements an agent that uses the Deep Q-Network algorithm for training.
    The agent is designed to solve the lunar lander gym environment.

    Attributes:

        config (dict): Dictionary that contains the configuration parameters of the agent.
        playback_buffer (numpy array): Replay buffer that stores past experiences of the agent.
        writer (Tensorboard writer): Tensorboard writer to log the training progress.
        q_model (torch.nn.Module): Q-network model used to approximate the Q-function.
        target_q_model (torch.nn.Module): Target Q-network model
        optimizer (torch.optim): PyTorch optimizer used to train the Q-network model.
        epsilon (float): Exploration rate of the agent.

    Methods:

        get_model(): Method that returns the Q-network model.
        learn(): Method that trains the DQN agent.

    """
    def __init__(self, config: Dict[str, Union[float, int]],
                 writer: SummaryWriter, model: torch.nn.Module) -> None:


        self.config = config
        self.playback_buffer = None
        self.writer = writer
        self.q_model = model(
            self.config['observation_space'],
            self.config['action_space']).to(DEVICE)
        self.target_q_model = model(
            self.config['observation_space'],
            self.config['action_space']).to(DEVICE)
        self.target_q_model.load_state_dict(self.q_model.state_dict())
        self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=0.0001)

        self.epsilon = 1.0

    def _e_greedy_action(self, state: ndarray) -> Union[int, int64]:
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.config['action_space'])
        else:
            action = np.argmax(predict(self.q_model, state, DEVICE))
        return action

    def _get_sample_batch(self, steps: int) -> ndarray:
        low_offset = steps \
                if steps < self.config['playback_buffer_size'] \
                else self.config['playback_buffer_size']
        sample_indexes = np.random.randint(
            (self.playback_buffer.shape[0] - low_offset),
            self.playback_buffer.shape[0],
            size=self.config['playback_sample_size'])
        sample_batch = np.copy(self.playback_buffer[sample_indexes])
        return sample_batch

    def _update_network(self, sample_batch: ndarray) -> None:

        current_states = np.stack(sample_batch[:, CURRENT_STATE_INDEX])
        rewards = np.vstack(sample_batch[:, REWARD_INDEX])
        actions = np.vstack(sample_batch[:, ACTION_INDEX]).T[0]
        next_states = np.stack(sample_batch[:, NEXT_STATE_INDEX])

        target_predictions = predict(
            self.target_q_model, next_states, DEVICE)
        max_actions = target_predictions.max(1)

        action_values = np.add(
            rewards.T, np.multiply(
                GAMMA, max_actions))[0]

        for sample_index, sample in enumerate(sample_batch):
            if sample[DONE_INDEX]:
                action_values[sample_index] = sample_batch[sample_index][
                    REWARD_INDEX]

        model_predictons = predict(self.q_model, current_states, DEVICE)
        model_predictons[range(len(sample_batch)), actions] = action_values

        data = {'features':current_states, 'target':model_predictons}

        train(
            data,
            self.q_model,
            self.optimizer,
            torch.nn.MSELoss(),
            DEVICE)

    def get_model(self):
        """
        Returns the Q-network model.

        Returns:
        --------
        torch.Tensor
            Q-network model.
        """
        return self.q_model

    def learn(self, env: gym.Env) -> None:
        """
        Train DQN agent of gym environment.

        Parameters:
        -----------
        env : gym environment
        """

        steps = 0

        average_episode_reward = deque(maxlen=100)
        top_avg_reward = 0

        for episode in range(self.config['num_episodes']):

            state_current = env.reset()

            total_episode_rewards = 0.0
            frames = 0

            while True:
                steps += 1
                frames += 1

                action = self._e_greedy_action(state_current)
                next_state, reward, done, _ = env.step(action)

                total_episode_rewards += reward

                if self.playback_buffer is None:
                    self.playback_buffer = np.array(
                        [state_current, action, reward, next_state, done],
                        dtype=object)
                    self.playback_buffer = np.vstack(
                        [self.playback_buffer] * self.config['playback_buffer_size'])
                else:
                    self.playback_buffer = np.vstack((self.playback_buffer, np.array(
                        [state_current, action, reward, next_state, done], dtype=object)))

                while self.playback_buffer.shape[0] > self.config['playback_buffer_size']:
                    self.playback_buffer = self.playback_buffer[1:]

                if ((steps > self.config['playback_sample_size'])
                        and (steps % self.config['dqn_train_rate'] == 0)):
                    self._update_network(self._get_sample_batch(steps))

                state_current = next_state

                if (steps % self.config['target_network_update_rate']) == 0:
                    self.target_q_model.load_state_dict(
                        self.q_model.state_dict())

                if done:
                    average_episode_reward.append(total_episode_rewards)

                    if self.epsilon > 0.1:
                        self.epsilon = self.epsilon * self.config['epsilon_reduction']

                    avg_reward = np.average(average_episode_reward)

                    if ((len(average_episode_reward) == 1)
                            or (avg_reward > top_avg_reward)):
                        top_avg_reward = avg_reward
                        torch.save(self.q_model.state_dict(),
                                   f"./model-pb_buff_sz-{self.config['playback_buffer_size']}-"\
                                   f"pb_sample_size-{self.config['playback_sample_size']}-"\
                                   f"target_network_update-"
                                   f"{self.config['target_network_update_rate']}.pth")

                    print(
                        f'EP:{episode} AVG REWARD:{avg_reward} EPSILON:{self.epsilon} FR:{frames}')

                    try:
                        self.writer.add_scalar(
                            'avg_reward', avg_reward, episode)
                    except AttributeError:
                        pass

                    report_tune(avg_reward)

                    break

        env.close()
