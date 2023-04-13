"""
This module provides functions for running and training a Deep Q-Network (DQN) agent.

Functions:
- run_agent(env, q_model, num_episodes):
    Runs the DQN agent in the specified environment for a given number of episodes.
    The agent uses the provided q_model to make predictions and select actions at each time step.
    It prints the total episode reward, epsilon value, and number of frames for each episode.

- train_dqn(config):
    Trains the DQN agent with the specified configuration.
    The configuration is a dictionary containing hyperparameters for the DQN agent.

- dqn_runner(model, env):
    Enables the DQN agent to be ran in different modes including: 'tune', 'train', or 'run'.
    It takes a model and an environment as input.
    It uses argparse for selecting the mode and loading training configuration files.
    It calls the corresponding functions run_agent() or train_dqn() depending on the selected mode.

- main():
    Entry point for running the lunar lander DQN application.

"""

import argparse
from typing import Dict, Union
import sys
import numpy as np
import gym
from ray import tune
from ray.tune import CLIReporter
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from lunar_lander_dqn.dqn_agent import AgentDQN
from lunar_lander_dqn.lunar_lander_net import LunarLanderNet
from lunar_lander_dqn.nn_utils import predict


def run_agent(env, q_model: torch.nn.Module, num_episodes: int) -> None:
    """
    Runs the agent in the environment for a specified number of episodes.

    Args:
        env: The environment to run the agent in.
        q_model (torch.Tensor): The Q-network model for the agent.
        num_episodes (int): The number of episodes to run the agent.

    Returns:
        None
    """

    for episode in range(num_episodes):
        state_current = env.reset()

        if not isinstance(state_current, np.ndarray):
            state_current = np.array([state_current])

        total_episode_rewards = 0
        frames = 0

        while True:
            env.render()
            frames += 1

            action = np.argmax(predict(q_model, state_current, 'cpu'))
            next_state, reward, done, _ = env.step(action)
            total_episode_rewards = total_episode_rewards + reward
            state_current = next_state
            if not isinstance(state_current, np.ndarray):
                state_current = np.array([state_current])

            if done:
                print(
                    f'EPISODE: {episode} EPISODE REWARD: {total_episode_rewards} \
                            EPSILON: {0} FRAMES: {frames}')
                break

    env.close()


def train_dqn(config: Dict[str, Union[float, int]]) -> None:

    """
    Trains the DQN agent with the specified configuration.

    Args:
        config (dict): The configuration (hyper-parameters) dictionary for training the DQN agent.

    Returns:
        None
    """

    env = config['env']
    model = config['model']

    config['action_space'] = env.action_space.n
    config['observation_space'] = env.observation_space

    try:
        writer = config['writer']
    except KeyError:
        writer = None

    agent = AgentDQN(
        config,
        writer,
        model,
    )

    agent.learn(env)


def dqn_runner(model: torch.nn.Module, env: gym.Env) -> None:
    """
    Main function for running the DQN agent with different modes.
    Expects Q-Network Model and environment for agent.

    Args:
        model: The DQN model for the agent.
        env: The environment for the agent.

    Returns:
        None
    """

    parser = argparse.ArgumentParser()

    mx_group = parser.add_mutually_exclusive_group()
    mx_group.add_argument('--tune', default='None', help='Path to config file required', type=str)
    mx_group.add_argument('--train', default='None', help='Path to config file required', type=str)
    mx_group.add_argument('--run', default='None', help='Path to model file required', type=str)

    parser.add_argument('--num_episodes', required=True, type=int)

    args = parser.parse_args()

    if args.tune != 'None':

        with open(args.tune, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)

        try:
            config['env'] = env
            config['model'] = model

            config['num_episodes'] = args.num_episodes
            config['playback_buffer_size'] = tune.choice(
                config['playback_buffer_size'])
            config['playback_sample_size'] = tune.choice(
                config['playback_sample_size'])
            config['target_network_update_rate'] = tune.choice(
                config['target_network_update_rate'])
            num_samples = config.pop('num_samples')

        except KeyError:
            print('Improperly formatted config file')
            sys.exit(-1)

        reporter = CLIReporter(metric_columns=["reward", "training_iteration"])

        tune.run(
            train_dqn,
            name='dqn-tune',
            local_dir='data',
            config=config,
            num_samples=num_samples,
            stop={'training_iteration': args.num_episodes},
            progress_reporter=reporter)

    elif args.train != 'None':

        with open(args.train, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)

        try:
            writer = SummaryWriter(
                './{}/dqn-playback_buff_sz-{}-'\
                'playback_sample_size-{}-target_network_update-{}'.format(
                    'data',
                    config['playback_buffer_size'],
                    config['playback_sample_size'],
                    config['target_network_update_rate']),
                flush_secs=1)

            config["writer"] = writer

            config['env'] = env
            config['model'] = model
            config["num_episodes"] = args.num_episodes
        except KeyError:
            print('Improperly formatted config file')
            sys.exit(-1)

        train_dqn(config)

    elif args.run != 'None':

        q_model = model(env.observation_space, env.action_space.n)
        q_model.load_state_dict(torch.load(args.run))
        run_agent(env, q_model, args.num_episodes)

    else:
        print('Must specify the desired action of the agent: tune, train, or run.')
        print('This request must me accompanied by a path to a config or model file')

def main() -> None:
    """main"""
    env = gym.make('LunarLander-v2')
    dqn_runner(LunarLanderNet, env)


if __name__ == "__main__":
    main()
