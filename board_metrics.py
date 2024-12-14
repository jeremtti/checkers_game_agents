from board import Board

def basic_score(board : Board):
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