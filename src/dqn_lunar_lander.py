import argparse
from collections import deque
import gym
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import random

def model_init(observation_space, action_space):
    model = Sequential()
    model.add(Dense(70, input_dim=observation_space, activation='relu'))
    model.add(Dense(70, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.0001))
    return model

class LunarLanderDQN:
    def __init__ (self):
        ACTION_SPACE = 4
        OBSERVATION_SPACE = 8
        MAX_CAP = 4500
        self.action_space = ACTION_SPACE
        self.observation_space = OBSERVATION_SPACE
        self.playback_buffer = deque(maxlen=MAX_CAP)
        self.q_model = model_init(OBSERVATION_SPACE, ACTION_SPACE)
        self.target_q_model = model_init(OBSERVATION_SPACE, ACTION_SPACE)
        self.target_q_model.set_weights(self.q_model.get_weights())
        self.epsilon = 0.99
        self.epsilon_reduction = 0.999
        self.num_episodes = 3000
        self.gamma = 0.99

    def e_greedy_action(self, state_current, epsilon):
        if np.random.random() < epsilon:
            action = np.random.randint(self.action_space)
        else:
            action = np.argmax(self.q_model.predict(state_current))
        return action

    def dqn_update(self, sample_batch):

        CURRENT_STATE_INDEX = 0
        ACTION_INDEX = 1
        REWARD_INDEX = 2
        NEXT_STATE_INDEX = 3
        DONE_INDEX = 4
        NUM_EPOCHS = 1

        current_states = np.vstack(sample_batch[:, CURRENT_STATE_INDEX])
        next_states = np.vstack(sample_batch[:, NEXT_STATE_INDEX])
        rewards = np.vstack(sample_batch[:, REWARD_INDEX])
        actions = np.vstack(sample_batch[:, ACTION_INDEX]).T[0]

        target_predictions = self.target_q_model.predict(next_states)

        max_actions = target_predictions.max(1)

        performance_values = np.add(rewards.T, np.multiply(self.gamma, max_actions))[0]

        for sample_index in range(len(sample_batch)):
            if sample_batch[sample_index][DONE_INDEX]:
                performance_values[sample_index] = sample_batch[sample_index][REWARD_INDEX]

        model_predictons = self.q_model.predict(current_states)

        model_predictons[range(len(sample_batch)), actions] = performance_values

        self.q_model.fit(current_states, model_predictons, epochs=NUM_EPOCHS, verbose=False)

    def get_sample_batch(self, sample_batch_size):
        sample_batch = random.sample(self.playback_buffer, sample_batch_size)
        sample_batch = np.array(sample_batch)
        return sample_batch

    def dqn_learning(self):

        env = gym.make('LunarLander-v2')
        SAMPLE_BATCH_SIZE = 25
        TARGET_UPDATE = 200
        steps = 0
        average_episode_reward = deque(maxlen=100)

        for episode in range(self.num_episodes):
            state_current = np.array([env.reset()])
            total_episode_rewards = 0
            frames = 0

            while True:
                steps += 1
                frames += 1

                action = self.e_greedy_action(state_current, self.epsilon)

                next_state, reward, done, info = env.step(action)
                next_state = np.array([next_state])
                total_episode_rewards = total_episode_rewards + reward

                self.playback_buffer.append([state_current, action, reward, next_state, done])

                if len(self.playback_buffer) > SAMPLE_BATCH_SIZE:
                    sample_batch = self.get_sample_batch(SAMPLE_BATCH_SIZE)
                    self.dqn_update(sample_batch)

                state_current = next_state

                if steps % TARGET_UPDATE is 0:
                    self.target_q_model.set_weights(self.q_model.get_weights())

                if done:
                    average_episode_reward.append(total_episode_rewards)

                    if (self.epsilon > 0.1):
                        self.epsilon = self.epsilon*self.epsilon_reduction

                    print (f'EPISODE: {episode} EPISODE REWARD: {total_episode_rewards} AVERAGE REWARD: {np.average(average_episode_reward)} EPSILON: {self.epsilon} FRAMES: {frames}')
                    break

        return self.q_model

def dqn_train_lunar_lander():

    lunar_lander = LunarLanderDQN()

    q_model = lunar_lander.dqn_learning()
    return q_model

def dqn_run_lunar_lander(q_model=None, weights_file=None):

    NUM_EPISODES = 100
    ACTION_SPACE = 4
    OBSERVATION_SPACE = 8

    if weights_file is not None:
        q_model = model_init(OBSERVATION_SPACE, ACTION_SPACE)
        q_model.load_weights(weights_file)

    env = gym.make('LunarLander-v2')

    for episode in range(NUM_EPISODES):
        state_current = np.array([env.reset()])
        total_episode_rewards = 0
        frames = 0



        while True:
            env.render()

            frames += 1

            action = np.argmax(q_model.predict(state_current))
            next_state, reward, done, info = env.step(action)
            total_episode_rewards = total_episode_rewards + reward
            next_state = np.array([next_state])
            state_current = next_state

            if done:
                print (f'EPISODE: {episode} EPISODE REWARD: {total_episode_rewards} EPSILON: {0} FRAMES: {frames}')
                break

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("training")
    parser.add_argument("file_path")
    args = parser.parse_args()

    if args.training in ['None', 'True']:
        dqn_train_lunar_lander()
    elif args.training in ['False']:

        if args.file_path in ['None']:
            print('No file path specifed!')
            exit(0)

        dqn_run_lunar_lander(weights_file=args.file_path)
        exit(0)
    else:
        print('Unknown Error has occured. Ensure that arguments were entered correctly.')
