from abstract_agent import Agent
import board_metrics as bm
from board import Board
import numpy as np
import copy


class OneStepAgent(Agent):
    """
    Class that represents a simple agent that looks one-step in the future.
    """

    def move(self, board : Board):
        """
        Method that returns the move of the agent.
        The agent wants to maximize the score of the board after one move.
        """
        allowed_moves = board.get_allowed_moves()
        scores = []
        for move in allowed_moves:
            board_copy = copy.deepcopy(board)
            board_copy.move(move)
            scores.append(bm.basic_score(board_copy))
        return allowed_moves[np.argmax(scores)]