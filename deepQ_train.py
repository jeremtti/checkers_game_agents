from deepQ_utils import QNetwork, EpsilonScheduler, MinimumExponentialLR, initialize_q_network
from board import Board
from deepQ_agent import DeepQAgent
from abstract_agent import Agent
from game_runner import GameRunner
import numpy as np
import torch
import matplotlib.pyplot as plt
from tree_agent import DecisionTreeAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

side = 10

n_epochs = 10

gm = GameRunner()

q_network = initialize_q_network(side).to(device)
#q_network = torch.load("q_network.pth")
agent = DeepQAgent(q_network)
random_agent = Agent()

print("Before training")
A, B, t = gm.compare_agents(random_agent, agent, 100)
print(f"Agent won {A} games, Random agent won {B} games")


success_rates_final = []

for epoch in range(n_epochs):
    print(f"Final reward Epoch {epoch+1}")

    q_network = agent.q_network
    n_games = 100
    gamma = 0.9
    epsilon_start = 0.7
    epsilon_min = 0.4
    epsilon_decay = 1 - 1e-4
    epsilon_scheduler = EpsilonScheduler(epsilon_start, epsilon_min, epsilon_decay)
    lr = 0.01
    optimizer = torch.optim.Adam(q_network.parameters(), lr=lr)
    lr_scheduler = MinimumExponentialLR(optimizer, 0.9, min_lr=0.0001)
    loss_fn = torch.nn.MSELoss()

    print('Training...')

    """agent.train_imediate_reward(
        n_games,
        gamma,
        epsilon_scheduler,
        optimizer,
        lr_scheduler,
        loss_fn,
        device
    )"""

    agent.train_final_reward(
        n_games,
        gamma,
        epsilon_scheduler,
        optimizer,
        lr_scheduler,
        loss_fn,
        device
    )

    # Test the agent against a random agent
    print('Validation...')
    A, B, t = gm.compare_agents(agent, random_agent, 100)
    print(f"Agent won {A} games, Random agent won {B} games")
    success_rates_final.append(A)


    torch.save(agent.q_network, f"q_network_final{str(epoch)}.pth")

success_rates_imediate = []

for epoch in range(n_epochs):
    print(f"Immediate reward Epoch {epoch+1}")

    q_network = agent.q_network
    n_games = 100
    gamma = 0.9
    epsilon_start = 0.7
    epsilon_min = 0.4
    epsilon_decay = 1 - 1e-4
    epsilon_scheduler = EpsilonScheduler(epsilon_start, epsilon_min, epsilon_decay)
    lr = 0.01
    optimizer = torch.optim.Adam(q_network.parameters(), lr=lr)
    lr_scheduler = MinimumExponentialLR(optimizer, 0.9, min_lr=0.0001)
    loss_fn = torch.nn.MSELoss()

    print('Training...')

    agent.train_imediate_reward(
        n_games,
        gamma,
        epsilon_scheduler,
        optimizer,
        lr_scheduler,
        loss_fn,
        device
    )
    """
    agent.train_final_reward(
        n_games,
        gamma,
        epsilon_scheduler,
        optimizer,
        lr_scheduler,
        loss_fn,
        device
    )"""

    # Test the agent against a random agent
    print('Validation...')
    A, B, t = gm.compare_agents(agent, random_agent, 100)
    print(f"Agent won {A} games, Random agent won {B} games")
    success_rates_imediate.append(A)


    torch.save(agent.q_network, f"q_network_immediate{str(epoch)}.pth")


plt.plot(success_rates_imediate, label="turn reward")
plt.plot(success_rates_final, label="final reward")
plt.xlabel("Training epoch")
plt.ylabel("Success rate")
plt.title("Training success rate")
plt.legend()
plt.show()