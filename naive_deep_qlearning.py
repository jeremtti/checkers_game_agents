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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_board(envi):
    plt.figure()
    plt.imshow(envi.get_image())
    
    plt_img = plt.gcf()
    plt_img.canvas.draw()
    pil_img = Image.frombytes('RGB', plt_img.canvas.get_width_height(), plt_img.canvas.tostring_rgb())
    plt.close()
    
    return pil_img



winning_reward = 500



def board_metric(envi):
    """
    Simple evaluation function for the checkers game.
    """
    b = envi.board
    
    men1 = np.sum((b==1))
    men2 = np.sum((b==-1))
    kings1 = np.sum((b==2))
    kings2 = np.sum((b==-2))
    
    if men2+kings2 == 0:
        return winning_reward
    elif men1+kings1 == 0:
        return -winning_reward
        
    men = men1-men2
    kings = kings1-kings2

    potential_moves_white = len(envi.get_allowed_moves())
    envi.transpose()
    potential_moves_black = len(envi.get_allowed_moves())
    envi.transpose()

    potential = potential_moves_white - potential_moves_black

    score = men + 5*kings #+ potential
    
    return score



def state_to_first_layer(state):
    output = np.zeros(5*len(state))
    for i in range(len(state)):
        output[5*i] = (state[i]==0)
        output[5*i+1] = (state[i]==1)
        output[5*i+2] = (state[i]==-1)
        output[5*i+3] = (state[i]==2)
        output[5*i+4] = (state[i]==-2)
    return output



class naive_QNetwork(torch.nn.Module):

    def __init__(self, n_observations: int, n_actions: int, nn_l1: int, nn_l2: int):
        super(naive_QNetwork, self).__init__()
        self.layer1 = torch.nn.Linear(n_observations, nn_l1)
        self.layer2 = torch.nn.Linear(nn_l1, nn_l2)
        self.layer3 = torch.nn.Linear(nn_l2, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = self.layer3(x)

        return x



class EpsilonGreedy:

    def __init__(self,
                 epsilon_start: float,
                 epsilon_min: float,
                 epsilon_decay:float,
                 env: Board,
                 q_network: torch.nn.Module):
        
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.env = env
        self.q_network = q_network

    def __call__(self, state: np.ndarray) -> np.int64:
        
        real = np.random.rand()
        if real<1-self.epsilon:
            q_values = self.q_network(torch.tensor(state_to_first_layer(state), dtype=torch.float32, device=device).unsqueeze(0))
            action = qvalues_to_best_possible_move(self.env, q_values)
           
        else:
            action = random.choice(self.env.get_allowed_moves())

        return action

    def decay_epsilon(self):
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)



class MinimumExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer: torch.optim.Optimizer, lr_decay: float, last_epoch: int = -1, min_lr: float = 1e-6):
        
        self.min_lr = min_lr
        super().__init__(optimizer, lr_decay, last_epoch=-1)

    def get_lr(self):
        
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
            for base_lr in self.base_lrs
        ]



def qvalues_to_best_possible_move(envi, q_values):
    """
    Convert the Q-values to the best possible move.
    """
    
    allowed_moves = envi.get_allowed_moves()
    
    q_values = q_values.detach().cpu().numpy()[0]
    
    action = allowed_moves[0][0]
    jumped = allowed_moves[0][1]
    for i in range(len(allowed_moves)):
        if q_values[allowed_moves[i][0][0]*envi.size+allowed_moves[i][0][1]] > q_values[action[0]*envi.size+action[1]]:
            action = allowed_moves[i][0]
            jumped = allowed_moves[i][1]

    return (action, jumped)



def train_naive_agent_turn_reward(envi: Board,
                      q_network: torch.nn.Module,
                      optimizer: torch.optim.Optimizer,
                      loss_fn: Callable,
                      device: torch.device,
                      lr_scheduler: _LRScheduler,
                      epsilon_greedy: EpsilonGreedy,
                      num_episodes: int,
                      gamma: float,
                      render = True) -> List[float]:
    
    episode_reward_dict = {0: [], 1: []}

    for episode_index in range(1, num_episodes):
        # print(f"Episode {episode_index}/{num_episodes}")
        
        envi.reset()
        state = envi.board
        done = False
        episode_reward = [0, 0]
        player = 0
        action = None
        
        images = []

        for t in itertools.count():
            
            #print(f"Player {player} move")
            
            action = epsilon_greedy.__call__(state)
            #print(f"Action: {action}")
            envi.move(action)
            
            reward = board_metric(envi)
            episode_reward[player] += reward
            #print(f"Episode counting reward: {episode_reward}")
            
            envi.transpose()
            sprime = envi.board
            player = 1 - player
            done = envi.is_final()
            
            predictions = q_network(torch.tensor(state_to_first_layer(state), dtype=torch.float32, device=device).unsqueeze(0))[0]
            prediction = predictions[action[0][0]*envi.size+action[0][1]]
            
            future_predictions = q_network(torch.tensor(state_to_first_layer(sprime), dtype=torch.float32, device=device).unsqueeze(0))[0]
            future_predictions = future_predictions.detach().cpu().numpy()
            
            future_prediction = future_predictions[0]
            allowed_moves = envi.get_allowed_moves()
            for allowed_move in allowed_moves:
                if future_predictions[allowed_move[0][0]*envi.size+allowed_move[0][1]] > future_prediction:
                    future_prediction = future_predictions[allowed_move[0][0]*envi.size+allowed_move[0][1]] 
            target = reward + gamma * future_prediction
            target = torch.tensor(target, dtype=torch.float32, device=device)
            loss = loss_fn(prediction, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            state = sprime
            
            if render:
                if player==0:
                    images.append(show_board(envi))
                else:
                    envi.transpose()
                    images.append(show_board(envi))
                    envi.transpose()
            
            if done:
                break
        
        episode_reward_dict[0].append(episode_reward[0])
        episode_reward_dict[1].append(episode_reward[1])
        epsilon_greedy.decay_epsilon()
        # Create a GIF from the images
        if render:
            images[0].save(f'train_episode_{episode_index}.gif',
                save_all=True, append_images=images[1:], optimize=False, duration=500, loop=0)
    
    return episode_reward_dict



def test_q_network_agent(envi, q_network: torch.nn.Module, num_episode: int = 1, render: bool = True):
    
    episode_reward_dict = {0: [], 1: []}

    for episode_id in range(num_episode):

        envi.reset()
        state = envi.board
        done = False
        episode_reward = [0, 0]
        player = 0
        action = None
        
        images = []

        while not done:
            
            state_tensor = torch.tensor(state_to_first_layer(state), dtype=torch.float32, device=device).unsqueeze(0)
            q_values = q_network(state_tensor)
            action = qvalues_to_best_possible_move(envi, q_values)

            envi.move(action)
            
            reward = board_metric(envi)
            episode_reward[player] += reward
            
            envi.transpose()
            player = 1 - player
            
            state = envi.board
            done = envi.is_final()
            
            if player==0:
                images.append(show_board(envi))
            else:
                envi.transpose()
                images.append(show_board(envi))
                envi.transpose()
            
        # Create a GIF from the images
        images[0].save(f'test_episode_{episode_id}.gif',
            save_all=True, append_images=images[1:], optimize=False, duration=500, loop=0)

        episode_reward_dict[0].append(episode_reward[0])
        episode_reward_dict[1].append(episode_reward[1])

    return episode_reward_dict


def train_naive_agent_final_reward(envi: Board,
                      q_network: torch.nn.Module,
                      optimizer: torch.optim.Optimizer,
                      loss_fn: Callable,
                      device: torch.device,
                      lr_scheduler: _LRScheduler,
                      epsilon_greedy: EpsilonGreedy,
                      num_episodes: int,
                      gamma: float,
                      render = True) -> List[float]:
    
    episode_reward_dict = {0: [], 1: []}

    for episode_index in range(1, num_episodes):
        #print(f"Episode {episode_index}/{num_episodes}")
        
        envi.reset()
        state = envi.board
        done = False
        episode_reward = [0, 0]
        player = 0
        action = None
        
        images = []
        game_predictions = []
        game_futur_predictions= []
        for t in itertools.count():
            
            #print(f"Player {player} move")
            
            action = epsilon_greedy.__call__(state)
            #print(f"Action: {action}")
            envi.move(action)
            
            
            envi.transpose()
            sprime = envi.board
            player = 1 - player
            done = envi.is_final()
            
            predictions = q_network(torch.tensor(state_to_first_layer(state), dtype=torch.float32, device=device).unsqueeze(0))[0]
            prediction = predictions[action[0][0]*envi.size+action[0][1]]
            

            future_predictions = q_network(torch.tensor(state_to_first_layer(sprime), dtype=torch.float32, device=device).unsqueeze(0))[0]
            future_predictions = future_predictions.detach().cpu().numpy()
            
            
            future_prediction = future_predictions[0]
            allowed_moves = envi.get_allowed_moves()
            for allowed_move in allowed_moves:
                if future_predictions[allowed_move[0][0]*envi.size+allowed_move[0][1]] > future_prediction:
                    future_prediction = future_predictions[allowed_move[0][0]*envi.size+allowed_move[0][1]]

            game_predictions.append(prediction)
            game_futur_predictions.append(torch.tensor(future_prediction, dtype=torch.float32, device=device))

            state = sprime
            
            if render:
                if player==0:
                    images.append(show_board(envi))
                else:
                    envi.transpose()
                    images.append(show_board(envi))
                    envi.transpose()
            
            if done:
                break
        
        torch.autograd.set_detect_anomaly(True)
        reward = 1 / len(game_predictions)
        for t in range(len(game_predictions)):
            optimizer.zero_grad()
            futur_prediction = game_futur_predictions[-t-1]
            prediction = game_predictions[-t-1]
            target = torch.tensor(reward + gamma * futur_prediction, dtype=torch.float32, device=device)
            loss = loss_fn(prediction, target)
            
            
            loss.backward(retain_graph=True)
            reward = - reward
        optimizer.step()
        lr_scheduler.step()
            

            
            
        
        epsilon_greedy.decay_epsilon()
        # Create a GIF from the images
        if render:
            images[0].save(f'train_episode_{episode_index}.gif',
                save_all=True, append_images=images[1:], optimize=False, duration=500, loop=0)
    
    return episode_reward_dict
