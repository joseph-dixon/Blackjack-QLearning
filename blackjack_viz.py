from matplotlib import pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', None)
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d

## Read in Q Table, rename columns
# probs = pd.read_csv('/Users/josephdixon/Desktop/Grad School/MSDS680/Week 8/q_table.csv',index_col = False)
# probs.columns = ['prob_stay','prob_hit']
#
# # Add columns for each element in Observation
# player_hand = []
# dealer_hand = []
# is_ace = []
# for i in range(4,22):
#     for j in range(1,11):
#         for a in range(0,2):
#             player_hand.append(i)
#             dealer_hand.append(j)
#             is_ace.append(a)
# probs['player_hand'] = player_hand
# probs['dealer_hand'] = dealer_hand
# probs['is_ace'] = is_ace
# probs = probs[['player_hand','dealer_hand','is_ace','prob_stay','prob_hit']]
#
# # Remove game scenerios that didn't occur
# probs_stripped = probs.drop(probs[probs.prob_stay == 0].index)
# probs_stripped = probs.drop(probs[probs.prob_hit == 0].index)
#
# # Aggregate by player_hand, mean
# player_hand_stripped = probs_stripped[['player_hand','prob_stay','prob_hit']]
# player_hand_probs = player_hand_stripped.groupby('player_hand').agg('mean')
#
# # 2D plot
# sns.lineplot(data = player_hand_probs)
# plt.show()
#
# # Aggregate by dealer_hand, mean
# hand_stripped = probs[['player_hand','dealer_hand','prob_stay','prob_hit']]
# hand_probs = hand_stripped.groupby(['player_hand','dealer_hand'], as_index = False).agg('mean')
# player_hand_trunc = []
# dealer_hand_trunc = []
# for i in range(4,22):
#     for j in range(1,11):
#         player_hand_trunc.append(i)
#         dealer_hand_trunc.append(j)
# hand_probs['player_hand'] = player_hand_trunc
# hand_probs['dealer_hand'] = dealer_hand_trunc
# hand_probs_trunc = hand_probs[['player_hand','dealer_hand','prob_stay','prob_hit']]
#
#
# # 3D plot
# x = hand_probs_trunc['player_hand']
# y = hand_probs_trunc['dealer_hand']
# z_1 = hand_probs_trunc['prob_stay']
# z_2 = hand_probs_trunc['prob_hit']
# fig = plt.figure()
# fig.set_size_inches(8, 8)
# ax = plt.axes(projection='3d')
# ax.plot(x,y,z_1, label = 'Stay')
# ax.plot(x,y,z_2, label = 'Hit')
# ax.legend()
# plt.xlabel('Player Hand')
# plt.ylabel('Dealer Card')
# plt.show()

# Read in rewards over time
rand_rewards = pd.read_csv('/Users/josephdixon/Desktop/Grad School/MSDS680/Week 8/rand_reward_table.csv', index_col = False)
rewards = pd.read_csv('/Users/josephdixon/Desktop/Grad School/MSDS680/Week 8/reward_table.csv', index_col = False)
sns.lineplot(data = rand_rewards, x = 'rand_time_step', y = 'rand_rewards', label = 'random')
sns.lineplot(data = rewards, x = 'timestep', y = 'reward', label = 'agent')
plt.show()
