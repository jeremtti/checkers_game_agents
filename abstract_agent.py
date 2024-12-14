from environment import Environment
from board import Board
import random

class Agent:
    """
    Class that represents an agent.
    """

    def move(self, board : Board):
        """
        Method that returns the move of the agent.
        Random possible move is returne by default.
        """
        return random.choice(board.get_allowed_moves())