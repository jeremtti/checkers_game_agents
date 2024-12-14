from environment import Environment
from board import Board
import random
from naive_deep_qlearning import state_to_first_layer, qvalues_to_best_possible_move
import torch

class Q_learning_Agent:
    """
    Class that represents an agent.
    """
    def __init__(self, q_network_) -> None:
        self.q_network = q_network_
        pass
    def move(self, board : Board):
        """
        Method that returns the move of the agent.
        Random possible move is returne by default.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_tensor = torch.tensor(state_to_first_layer(board.board), dtype=torch.float32, device=device).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        action = qvalues_to_best_possible_move(board, q_values)
        return action