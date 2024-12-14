from abstract_agent import Agent
from board import Board
import numpy as np
import copy


class DecisionTreeAgent(Agent):
    """
    Class that represents a simple agent.
    """

    def __init__(self, nb_future=1, fear=2, hope=2) -> None:
        super().__init__()
        self.nb_future = nb_future
        self.fear = fear
        self.hope = hope
        self.victory = 1000


    def move(self, board : Board):
        """
        Method that returns the move of the agent.
        Explore all the possible futures with nb_future turns in advance and choose the best one.
        If nb_future is even, we want to maximize the score (white will be playing), if nb_future
        is not event, we want to minimize the score.
        """
        allowed_moves = board.get_allowed_moves()
        N = len(allowed_moves)
        all_scores = self.scores_for_first_move(board, self.nb_future)
        cumul_score = [0 for _ in range(N)]
        for i in range(N):
            scores = all_scores[i]
            cumul_score[i] = np.mean(scores) + self.hope * np.max(scores) - self.fear * np.min(scores)
        cumul_score = np.array(cumul_score)
        if self.nb_future % 2 == 0:
            return allowed_moves[np.argmax(cumul_score)]
        else:
            return allowed_moves[np.argmin(cumul_score)]


    def build_tree(self, board : Board, depth : int):
        """
        Method that builds the tree of possible futures.
        Each node is a move.
        Each leaf is a board.
        """
        if depth == 0:
            return board
        else:
            tree = []
            for move in board.get_allowed_moves():
                board_copy = copy.deepcopy(board)
                board_copy.move(move)
                board_copy.transpose()
                if board_copy.is_final():
                    tree.append("Victory")
                else:
                    tree.append(self.build_tree(board_copy, depth-1))
            return tree
        
    def analyse_tree(self, tree):
        if isinstance(tree, Board):
            board = tree
            return (0,0,[self.board_score(board)])
        else:
            if tree == "Victory":
                return (0,1,[])
            else:
                scores = []
                white_victories = 0
                black_victories = 0
                for node in tree:
                    w, b, s = self.analyse_tree(node)
                    white_victories += b
                    black_victories += w
                    scores += s
                return (white_victories, black_victories, scores)
        
    def scores_for_first_move(self, board : Board, depth : int):
        allowed_moves = board.get_allowed_moves()
        scores = [0 for _ in range(len(allowed_moves))]
        for i in range(len(allowed_moves)):
            board_copy = copy.deepcopy(board)
            board_copy.move(allowed_moves[i])
            board_copy.transpose()
            if board_copy.is_final():
                scores[i] = [self.victory]
            else:
                tree = self.build_tree(board_copy, depth-1)
                w, b, s = self.analyse_tree(tree)
                score = np.mean(s) + self.hope * np.max(s) - self.fear * np.min(s)
        return all_scores

    def board_score(self, board : Board):
        """
        Method that returns the score of the board.
        """
        white_pieces = 0
        white_queens = 0
        black_pieces = 0
        black_queens = 0
        for tile in board.board:
            if tile == -2:
                black_queens += 1
            if tile == -1:
                black_pieces += 1
            if tile == 1:
                white_pieces += 1
            if tile == 2:
                white_queens += 1

        potential_moves_white = len(board.get_allowed_moves())

        board.transpose()
        potential_moves_black = len(board.get_allowed_moves())
        board.transpose()

        score = white_pieces + 5 * white_queens - black_pieces - 5 * black_queens + potential_moves_white - potential_moves_black
        
        return score