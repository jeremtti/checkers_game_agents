import numpy as np

class Board:
    """
    Class to represent the board and its state.
    """

    def __init__(self, side=10):
        """
        Initializes board based on international checkers rules.
        Architecture is compatible with other side lengths.
        """

        assert side % 2 == 0, "Side length must be even."

        self.side = side
        self.half = int(side / 2)

        # Number of usable squares
        self.size = int(self.side * self.side / 2)
        nb_pieces = 2 * side

        # Registering only playable squares, 0 = empty, 1 = white, -1 = black, 2 = white queen, -2 = black queen.
        # Makes it easy to "flip" the board
        self.board = np.array([0 for i in range(self.size)]) 
        self.board[:nb_pieces] -= 1
        self.board[-nb_pieces:] += 1

    def transpose(self):
        """
        Transposes the board.
        """
        self.board = np.flip(self.board)
        self.board = -self.board
        return self
    
    def get_coordinates(self, tile):
        """
        Returns the coordinates of a square given its id.
        """
        y = tile // self.half
        x = 2 * (tile % self.half) + (1 - y % 2)
        return x, y
    
    def get_tile(self, coordinates):
        """
        Returns the id of a square given its coordinates.
        """
        x, y = coordinates
        return int((y * self.half) + (x // 2))

    def next(self, tile, direction):
        """
        Returns the id of the next square in a given direction.
        Returns -1 if the next square is out of the board.
        """
        dx, dy = {
            'UL': (-1, -1),
            'UR': (1, -1),
            'DL': (-1, 1),
            'DR': (1, 1)
        }[direction]

        x, y = self.get_coordinates(tile)
        xp, yp = x + dx, y + dy
        if xp >= 0 and xp < self.side and yp >= 0 and yp < self.side:
            return self.get_tile((xp, yp))
        else:
            return -1
    

    def get_sequences_from_tile_for_piece(self, tile, already_taken_pieces):
        """
        Returns the sequences (and its associates taken pieces) that can be made from a given tile.
        For a piece
        """
        
        sequences = [([tile], [])]

        # Basic moves if no pieces have been taken
        if already_taken_pieces == []:
            for direction in ['UL', 'UR']:
                next_tile = self.next(tile, direction)
                if next_tile != -1 and self.board[next_tile] == 0:
                    sequences.append(([tile, next_tile], []))

        # Taking moves
        for direction in ['UL', 'UR', 'DL', 'DR']:
            next_tile = self.next(tile, direction)
            final_tile = self.next(next_tile, direction)
            if next_tile != -1 and final_tile != -1 and self.board[next_tile] < 0 and self.board[final_tile] == 0 and next_tile not in already_taken_pieces:
                # recursion
                for sequence, taken_pieces in self.get_sequences_from_tile_for_piece(final_tile, already_taken_pieces + [next_tile]):
                    sequences.append(([tile] + sequence, [next_tile] + taken_pieces))
            
        return sequences
    
    def get_sequences_from_tile_for_queen(self, tile, already_taken_pieces):
        """
        Returns the sequences (and its associates taken pieces) that can be made from a given tile.
        For a queen
        """
        
        sequences = [([tile], [])]

        # Basic moves if no pieces have been taken
        if already_taken_pieces == []:
            for direction in ['UL', 'UR', 'DL', 'DR']:
                next_tile = self.next(tile, direction)
                while next_tile != -1 and self.board[next_tile] == 0:
                    sequences.append(([tile, next_tile], []))
                    next_tile = self.next(next_tile, direction)

        # Taking moves
        for direction in ['UL', 'UR', 'DL', 'DR']:
            next_tile = self.next(tile, direction)
            while next_tile != -1 and self.board[next_tile] == 0:
                next_tile = self.next(next_tile, direction)
            to_be_taken = next_tile
            right_after = self.next(to_be_taken, direction)
            if self.board[to_be_taken] < 0 and right_after != -1 and self.board[right_after] == 0 and to_be_taken not in already_taken_pieces:
                # We can take the piece
                final_tile = right_after
                while final_tile != -1 and self.board[final_tile] == 0:
                    # recursion
                    for sequence, taken_pieces in self.get_sequences_from_tile_for_queen(final_tile, already_taken_pieces + [to_be_taken]):
                        sequences.append(([tile] + sequence, [to_be_taken] + taken_pieces))
                    final_tile = self.next(final_tile, direction)
        
        return sequences
    
    def get_allowed_moves(self):
        """
        Returns the allowed moves for the current player.
        The current player is white (positive tiles) and the opponent is black (negative tiles).
        The current player sees the board so that its pieces are close to them (big values of tiles)
        """
        allowed_moves = []
        for tile, piece in enumerate(self.board):
            if piece <= 0:
                sequences = []
            if piece == 1:
                sequences = self.get_sequences_from_tile_for_piece(tile, [])
            if piece == 2:
                sequences = self.get_sequences_from_tile_for_queen(tile, [])
            for sequence, taken_pieces in sequences:
                if len(sequence) > 1:
                    allowed_moves.append((sequence, taken_pieces))
        if allowed_moves == []:
            return []
        else:
            max_taken_pieces = max([len(taken_pieces) for sequence, taken_pieces in allowed_moves])
            allowed_moves = [(sequence, taken_pieces) for sequence, taken_pieces in allowed_moves if len(taken_pieces) == max_taken_pieces]
            return allowed_moves
    
    def move(self, move):
        """
        Moves a piece on the board.
        """
        if move not in self.get_allowed_moves():
            raise ValueError(f"Move {move} is not allowed.")
        sequence, taken_pieces = move
        start, end = sequence[0], sequence[-1]
        self.board[end] = self.board[start] # moving the piece
        self.board[start] = 0 # emptying the start square
        for piece in taken_pieces: # for every taken piece
            self.board[piece] = 0 # removing the piece
        if end < self.half: # if we reached the end of the board
            self.board[end] = 2 # crowning the piece
        return "Move done."
    
    def get_image(self):
        """
        Creates board image suitable for matplotlib.pyplot.imshow
        """
        # White board
        image = np.ones((self.side, self.side, 3))

        for i in range(self.size):
            x, y = self.get_coordinates(i)
            state = self.board[i]

            if state == 0:
                image[y, x] = [0, 0, 0]
            elif state == -1:
                image[y, x] = [1.0, 0, 0]
            elif state == 1:
                image[y, x] = [0, 1.0, 0]
            elif state == -2:
                image[y, x] = [0.3, 0, 0]
            elif state == 2:
                image[y, x] = [0, 0.3, 0]
        return image
    
    def is_final(self):
        """
        Returns True if the game is over.
        """
        return len(self.get_allowed_moves()) == 0

    

