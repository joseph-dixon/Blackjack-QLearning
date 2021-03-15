import gym
import numpy as np
import random
import pandas as pd

env = gym.make('Blackjack-v0')

unique_states = []
for player_hand in range(4,22):
    for dealer_hand in range(1,11):
        for ace_present in range(0,2):
            identifier = str(player_hand) + '_' + str(dealer_hand) + '_' + str(ace_present)
            unique_states.append(identifier)

q_table = {}
for unique_state in unique_states:
    q_table[unique_state] = {}
    for hit_or_stay in range(0,2):
        q_table[unique_state][hit_or_stay] = 0

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

for i in range(1, 1000001):

    # Initialize environment, assign state code
    done = False
    state = env.reset()
    if state[2] == False:
        is_ace = 0
    else :
        is_ace = 1
    state_code = str(state[0]) + '_' + str(state[1]) + '_' + str(is_ace)

    # While game still active, select action and get next state, reward, new environment
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            to_consider = [q_table[state_code][0], q_table[state_code][1]]
            action = np.argmax(to_consider) # Exploit learned values

        next_state, reward, done, info = env.step(action)

        if next_state[0] < 21:
            if next_state[2] == False:
                next_is_ace = 0
            else :
                next_is_ace = 1
            next_state_code = str(next_state[0]) + '_' + str(next_state[1]) + '_' + str(next_is_ace)

            old_value = q_table[state_code][action]

            next_to_consider = [q_table[next_state_code][0], q_table[next_state_code][1]]
            next_max = np.max(next_to_consider)

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state_code][action] = new_value

        else:

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state_code][action] = new_value

print("Training finished.\n")
df = pd.DataFrame(q_table).T
df.to_csv('/Users/josephdixon/Desktop/Grad School/MSDS680/Week 8/q_table.csv', index = False)

# For plotting metrics
total_reward = 0
ts = 0
time_step = []
rewards = []

for i in range(1001):

    # Initialize environment, assign state code
    done = False
    state = env.reset()
    if state[2] == False:
        is_ace = 0
    else :
        is_ace = 1
    state_code = str(state[0]) + '_' + str(state[1]) + '_' + str(is_ace)

    # While game still active, select action and get next state, reward, new environment
    while not done:
        to_consider = [q_table[state_code][0], q_table[state_code][1]]
        if to_consider[0] == 0 or to_consider[1] == 0:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(to_consider) # Exploit learned values

        next_state, reward, done, info = env.step(action)
        total_reward += reward
        ts += 1
        time_step.append(ts)
        rewards.append(total_reward)

        if next_state[0] < 21:
            if next_state[2] == False:
                next_is_ace = 0
            else :
                next_is_ace = 1
            next_state_code = str(next_state[0]) + '_' + str(next_state[1]) + '_' + str(next_is_ace)
            next_to_consider = [q_table[next_state_code][0], q_table[next_state_code][1]]
            next_max = np.max(next_to_consider)

rand_reward = 0
rand_ts = 0
rand_time_step = []
rand_rewards = []

for i in range(1001):
    # Initialize environment, assign state code
    done = False
    state = env.reset()

    # While game still active, select action and get next state, reward, new environment
    while not done:
        action = env.action_space.sample() # Explore action space
        next_state, reward, done, info = env.step(action)
        rand_ts += 1
        rand_reward += reward
        rand_time_step.append(rand_ts)
        rand_rewards.append(rand_reward)
        if done:
            break


df2 = pd.DataFrame()
df3 = pd.DataFrame()
df2['timestep'] = time_step
df2['reward'] = rewards
df3['rand_time_step'] = rand_time_step
df3['rand_rewards'] = rand_rewards
df2.to_csv('/Users/josephdixon/Desktop/Grad School/MSDS680/Week 8/reward_table.csv', index = False)
df3.to_csv('/Users/josephdixon/Desktop/Grad School/MSDS680/Week 8/rand_reward_table.csv', index = False)
