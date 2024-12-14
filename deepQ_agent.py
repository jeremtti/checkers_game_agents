from abstract_agent import Agent
from deepQ_utils import QNetwork
from deepQ_utils import EpsilonScheduler
from torch.optim.lr_scheduler import _LRScheduler
from board import Board
import numpy as np
import torch
import board_metrics

class DeepQAgent(Agent):
    """
    Class that represents an agent.
    """
    def __init__(self, q_network_: QNetwork) -> None:
        self.q_network = q_network_
        pass

    def move(self, board : Board):
        """
        Method that returns the move of the agent.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        moves = board.get_allowed_moves()
        q_values = np.zeros(len(moves))
        for m in range(len(moves)):
            (sequence, taken) = moves[m]
            state_tensor = torch.tensor(self.to_first_layer(board.board, sequence[0], sequence[1]), dtype=torch.float32, device=device).unsqueeze(0)
            q_value = self.q_network(state_tensor)
            q_values[m] = q_value
        action = moves[np.argmax(q_values)]
        return action
    
    def move_random(self, board : Board):
        """
        Method that returns a random possible move.
        """
        moves = board.get_allowed_moves()
        return moves[np.random.randint(len(moves))]

    
    def to_first_layer(self, state: np.array, tile_1: int, tile_2: int) -> np.ndarray:
        """
        Method that converts the state aznd the action to the first layer of the neural network.
        """
        N = len(state)
        output = np.zeros(7*N)
        for i in range(N):
            output[5*i] = (state[i]==0)
            output[5*i+1] = (state[i]==1)
            output[5*i+2] = (state[i]==-1)
            output[5*i+3] = (state[i]==2)
            output[5*i+4] = (state[i]==-2)
        for i in range(N):
            output[5*N + i] = (i==tile_1)
        for i in range(N):
            output[6*N + i] = (i==tile_2)
        return output
    
    def train_imediate_reward(self, 
                              n_games: int, 
                              gamma: float, 
                              epsilon_scheduler: EpsilonScheduler,
                              optimizer: torch.optim.Optimizer,
                              lr_scheduler: _LRScheduler,
                              loss_fn: torch.nn.modules.loss._Loss,
                              device: torch.device
                              ):
        """
        Method that trains the agent on n_games.
        """
        for g in range(n_games):
            board = Board()
            total_loss = 0
            n_grad = 0
            while True:

                # Check if the game is over
                if board.is_final():
                    break

                # Choose the action
                if epsilon_scheduler.random_action():
                    action = self.move_random(board)
                else:
                    action = self.move(board)

                # Compute the q_value
                (sequence, taken) = action
                state_tensor = torch.tensor(self.to_first_layer(board.board, sequence[0], sequence[1]), dtype=torch.float32, device=device).unsqueeze(0)
                q_value = self.q_network(state_tensor)

                # Apply the action, get the reward and transpose the board
                board.move(action)
                reward = board_metrics.basic_score(board)
                board.transpose()
                state = board.board

                # Check if the game is over
                if board.is_final():
                    next_q_value = torch.tensor([[0]], dtype=torch.float32, device=device)
                
                    target = reward + gamma * next_q_value

                    # Compute the loss
                    loss = loss_fn(q_value, target)
                    total_loss += loss

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    n_grad += 1

                    # Go to the next game
                    break
                else:
                    # Compute the next q_value
                    action_adversary = self.move(board)
                    (sequence, taken) = action_adversary
                    next_state_tensor = torch.tensor(self.to_first_layer(board.board, sequence[0], sequence[1]), dtype=torch.float32, device=device).unsqueeze(0)
                    next_q_value = self.q_network(next_state_tensor)

                    # Adversary plays
                    #board.move(action_adversary)
                    #board.transpose()

                    # Compute the target
                    target = reward + gamma * next_q_value

                    # Compute the loss
                    loss = loss_fn(q_value, target)
                    total_loss += loss

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    n_grad += 1

        total_loss /= n_grad
        print(f'Loss: {total_loss}')
        print(f'Epsilon final: {epsilon_scheduler.epsilon}')


    def train_final_reward(
            self,
            n_games: int,
            gamma: float,
            epsilon_scheduler: EpsilonScheduler,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: _LRScheduler,
            loss_fn: torch.nn.modules.loss._Loss,
            device: torch.device
    ):
        """
        Method that trains the agent on n_games.
        """
        for g in range(n_games):

            board = Board()
            total_loss = 0

            inputs = [[], []]

            player = 0
            t = 0

            while True:

                # Check if the game is over
                if board.is_final():
                    victor = 1 - player
                    break

                # Choose the action
                if epsilon_scheduler.random_action():
                    action = self.move_random(board)
                else:
                    action = self.move(board)
                
                inputs[player].append(self.to_first_layer(board.board, action[0][0], action[0][1]))
                
                # Apply the action, get the reward and transpose the board
                board.move(action)
                board.transpose()
                player = 1 - player

                t += 1
        
            # Compute the rewards
            total_reward = 1000 / t

            # Compute the targets
            targets = [0, 0]
            targets[victor] = total_reward
            targets[1 - victor] = -total_reward

            # Compute the losses
            for i in range(2):
                inputs_tensor = torch.tensor(np.array(inputs[i]), dtype=torch.float32, device=device)
                targets_tensor = torch.tensor(np.array([[targets[i]]]*len(inputs[i])), dtype=torch.float32, device=device)
                q_values = self.q_network(inputs_tensor)
                loss = loss_fn(q_values, targets_tensor)
                total_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                   