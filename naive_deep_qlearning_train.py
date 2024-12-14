from board import Board
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import torch
from torch.optim.lr_scheduler import _LRScheduler
from PIL import Image, ImageDraw
import itertools
import pandas as pd
import random
from typing import List, Tuple, Deque, Optional, Callable
from naive_deep_qlearning import *


side = 10
size = int(side**2/2) # The size of the board
move_length = 2 # The length of a move. We do not consider longer moves for now

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUMBER_OF_TRAININGS = 1000
trains_result_dict = {0: [[], [], []], 1: [[], [], []]}

envi = Board(side=side)

for train_index in range(NUMBER_OF_TRAININGS):
    print(f"Training {train_index}/{NUMBER_OF_TRAININGS}")
    # Instantiate required objects

    q_network = naive_QNetwork(5*size, size**move_length, nn_l1=128, nn_l2=128).to(device)
    optimizer = torch.optim.AdamW(q_network.parameters(), lr=0.004, amsgrad=True)
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    lr_scheduler = MinimumExponentialLR(optimizer, lr_decay=0.97, min_lr=0.0001)
    loss_fn = torch.nn.MSELoss()
    epsilon_greedy = EpsilonGreedy(epsilon_start=0.82, epsilon_min=0.013, epsilon_decay=0.9675, env=envi, q_network=q_network)

    # Train the q-network
    episode_reward_dict = train_naive_agent_final_reward(envi,
                                            q_network,
                                            optimizer,
                                            loss_fn,
                                            device,
                                            lr_scheduler,
                                            epsilon_greedy,
                                            num_episodes=100,
                                            gamma=0.9,
                                            render=False)
    
    """episode_reward_dict = train_naive_agent_turn_reward(envi,
                                            q_network,
                                            optimizer,
                                            loss_fn,
                                            device,
                                            lr_scheduler,
                                            epsilon_greedy,
                                            num_episodes=100,
                                            gamma=0.9,
                                            render=False)"""
    
    trains_result_dict[0][0].extend(range(len(episode_reward_dict[0])))
    trains_result_dict[0][1].extend(episode_reward_dict[0])
    trains_result_dict[0][2].extend([train_index for _ in episode_reward_dict[0]])

    trains_result_dict[1][0].extend(range(len(episode_reward_dict[1])))
    trains_result_dict[1][1].extend(episode_reward_dict[1])
    trains_result_dict[1][2].extend([train_index for _ in episode_reward_dict[1]])

    torch.save(q_network, f"naive_q_network{train_index}.pth")

trains_result_df = []
trains_result_df.append(pd.DataFrame(np.array(trains_result_dict[0]).T, columns=["num_episodes", "mean_final_episode_reward", "training_index"]))
trains_result_df.append(pd.DataFrame(np.array(trains_result_dict[1]).T, columns=["num_episodes", "mean_final_episode_reward", "training_index"]))

# Save naive_trains_result_df[0] and [1] to csv
trains_result_df[0].to_csv('naive_trains_result_0.csv', index=False, sep=';')
trains_result_df[1].to_csv('naive_trains_result_1.csv', index=False, sep=';')

plt.figure()
plt.plot(trains_result_df[0]["num_episodes"], trains_result_df[0]["mean_final_episode_reward"])
plt.savefig("training_player0.pdf")
plt.close()
plt.figure()
plt.plot(trains_result_df[1]["num_episodes"], trains_result_df[1]["mean_final_episode_reward"])
plt.savefig("training_player1.pdf")
plt.close()

torch.save(q_network, "naive_q_network.pth")


# TESTING THE AGENT (against itself...)
q_network = torch.load("naive_q_network.pth").to(device)

envi = Board(side=side)

test_q_network_agent(envi, q_network, num_episode=1, render=True)
