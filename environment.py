import numpy as np

class Environment:
    # Playing on black board spaces, the spaces are numbered from top to bottom, left to right.
    # Top left corner is white.

    def __init__(self, side=10):
        """
        Initializes checker board based on international checkers rules.
        Architecture is compatible with other side lengths.
        """

        self.side = side
        self.half = int(self.side / 2)

        # Number of usable squares
        self.size = int(self.side * self.side / 2)

        # Registering only playable squares, 0 = empty, 1 = white, -1 = black, 2 = white queen, -2 = black queen.
        # Makes it easy to "flip" the board
        self.board = np.array([0 for i in range(self.size)]) 
        self.turn = 0
        self.board[0:20] -= 1
        self.board[30:50] += 1
    
    # Functions to convert between 2D space and np.array indices of squares
    def coordinates(self, square_id):
        y = square_id // self.half
        x = 2 * (square_id % self.half) + (1 - y % 2)
        return x, y
    
    def square_id(self, coordinates):
        x, y = coordinates
        return int((y * self.half) + (x // 2))
    
    def board_image(self):
        """
        Creates board image suitable for matplotlib.pyplot.imshow
        """
        # White board
        image = np.ones((self.side, self.side, 3))

        for i in range(self.size):
            x, y = self.coordinates(i)
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
    

    def is_valid(self, init_position, end_position, piece_type, already_jumped):
        """
        Checks whether a hypothetic move to end_position would be valid in a chain.
        Assumes piece_type gets to init_position after taking all pieces in already_jumped.
        """
        # Piece needs to land in empty spot
        if self.board[end_position] != 0:
            return False, None
        
        valid = False
        jumped_piece = None        
        x1, y1 = self.coordinates(init_position)
        x2, y2 = self.coordinates(end_position)

        # White piece
        if (self.turn%2 == 0) and (piece_type == 1): 
            # If move is first in the chain, one-diagonal, frontwards :
            # Basic move, valid
            if(self.board[init_position] == 1) and (y1 == y2 + 1) and (abs(x1 - x2) == 1):
                valid = True
                jumped_piece = None

            x_enemy = (x1 + x2) / 2
            y_enemy = (y1 + y2) / 2
            enemy_position = self.square_id((x_enemy, y_enemy))
            # If two-diagonal move with enemy piece in the middle, either first move or in chain :
            # Basic take, valid
            if((self.board[init_position] == 1) or len(already_jumped) > 0) and (abs(y1 - y2) == 2) and (abs(x1 - x2) == 2) and (self.board[enemy_position] < 0):
                valid = True
                jumped_piece = enemy_position

        # White queen
        if (self.turn%2 == 0) and (piece_type == 2):
            # Queen move
            # First diagonal
            if ((x1 - x2) == (y1 - y2)):
                min_x = min(x1,x2)
                min_y = min(y1,y2)
                diff = int(abs(x1 - x2))
                pieces_present = 0
                piece_pos = None

                for delta in range(1, diff):
                    square = self.square_id((min_x + delta, min_y + delta))
                    # Enemy piece, only one accepted
                    if self.board[square] < 0:
                        pieces_present += 1
                        piece_pos = square
                    # Friendly piece, none accepted
                    if self.board[square] > 0:
                        pieces_present += 2

                # If taking, and in chain or first move, valid
                if (pieces_present == 1) and (len(already_jumped) > 0 or (self.board[init_position] == 2)):
                    valid = True
                    jumped_piece = piece_pos
                # If moving in first move, valid
                elif (pieces_present == 0) and (self.board[init_position] == 2):
                    valid = True
                else:
                    valid = False

            # Second diagonal
            if ((x1 - x2) == (y2 - y1)):
                min_x = min(x1,x2)
                max_y = max(y1,y2)
                diff = int(abs(x1 - x2))
                pieces_present = 0
                piece_pos = None

                for delta in range(1, diff):
                    square = self.square_id((min_x + delta, max_y - delta))
                    # Enemy piece, only one accepted
                    if self.board[square] < 0:
                        pieces_present += 1
                        piece_pos = square
                    # Friendly piece, none accepted
                    if self.board[square] > 0:
                        pieces_present += 2

                # If taking, and in chain or first move, valid
                if (pieces_present == 1) and (len(already_jumped) > 0 or (self.board[init_position] == 2)):
                    valid = True
                    jumped_piece = piece_pos
                # If moving in first move, valid
                elif (pieces_present == 0) and (self.board[init_position] == 2):
                    valid = True
                else:
                    valid = False

        # Black piece
        if (self.turn%2 == 1) and (piece_type == -1): 
            # If move is first in the chain, one-diagonal, frontwards :
            # Basic move, valid
            if(self.board[init_position] == -1) and (y1 == y2 - 1) and (abs(x1 - x2) == 1):
                valid = True
                jumped_piece = None

            x_enemy = (x1 + x2) / 2
            y_enemy = (y1 + y2) / 2
            enemy_position = self.square_id((x_enemy, y_enemy))
            # If two-diagonal move with enemy piece in the middle, either first move or in chain :
            # Basic take, valid
            if((self.board[init_position] == -1) or len(already_jumped) > 0) and (abs(y1 - y2) == 2) and (abs(x1 - x2) == 2) and (self.board[enemy_position] > 0):
                valid = True
                jumped_piece = enemy_position

        # Black queen
        if (self.turn%2 == 1) and (piece_type == -2):
            # Queen move
            # First diagonal
            if ((x1 - x2) == (y1 - y2)):
                min_x = min(x1,x2)
                min_y = min(y1,y2)
                diff = int(abs(x1 - x2))
                pieces_present = 0
                piece_pos = None
                
                for delta in range(1, diff):
                    square = self.square_id((min_x + delta, min_y + delta))
                    # Enemy piece, only one accepted
                    if self.board[square] > 0:
                        pieces_present += 1
                        piece_pos = square
                    # Friendly piece, none accepted
                    if self.board[square] < 0:
                        pieces_present += 2

                # If taking, and, in chain or first move, valid
                if (pieces_present == 1) and (len(already_jumped) > 0 or (self.board[init_position] == -2)):
                    valid = True
                    jumped_piece = piece_pos
                # If moving in first move, valid
                elif (pieces_present == 0) and (self.board[init_position] == -2):
                    valid = True
                else:
                    valid = False

            # Second diagonal
            if ((x1 - x2) == (y2 - y1)):
                min_x = min(x1,x2)
                max_y = max(y1,y2)
                diff = int(abs(x1 - x2))
                pieces_present = 0
                piece_pos = None

                for delta in range(1, diff):
                    square = self.square_id((min_x + delta, max_y - delta))
                    # Enemy piece, only one accepted
                    if self.board[square] > 0:
                        pieces_present += 1
                        piece_pos = square
                    # Friendly piece, none accepted
                    if self.board[square] < 0:
                        pieces_present += 2

                # If taking, and in chain or first move, valid
                if (pieces_present == 1) and (len(already_jumped) > 0 or (self.board[init_position] == -2)):
                    valid = True
                    jumped_piece = piece_pos
                # If moving in first move, valid
                elif (pieces_present == 0) and (self.board[init_position] == -2):
                    valid = True
                else:
                    valid = False

        # To avoid double captures, we need to account for pieces already taken : already_jumped.
        if jumped_piece in already_jumped:
            return False, None
        
        return valid, jumped_piece
     
    def move(self, square_sequence):
        """
        Checks if a move is valid and executes it.
        """
        start_position = square_sequence[0]
        start_state = self.board[start_position]

        if start_state == 0:
            return "Invalid move"
        
        possibilities = self.possible_moves(self.turn%2)
        paths, takes = zip(*possibilities)

        if square_sequence not in paths:
            return "Invalid move"
        
        path_id = paths.index(square_sequence)
        jumped = takes[path_id]
        
        # Place piece at the last square of the move
        self.board[square_sequence[-1]] = start_state
        self.board[start_position] = 0

        # Remove pieces taken
        for position in jumped:
            self.board[position] = 0

        # Promote white pieces on top row
        for square in range(self.half):
            if self.board[square] == 1:
                self.board[square] = 2

        # Promote black pieces on bottom row
        for square in range(self.size - self.half, self.size):
            if self.board[square] == -1:
                self.board[square] = -2    

        # Next turn
        self.turn = self.turn + 1

        return "Valid move"
    
    def get_reverse(self):
        """
        Returns environment viewed from the opponents side.
        """
        reverse = Environment()
        reverse.board = - self.board[::-1]
        return reverse
    
    def possible_moves(self, player):
        """
        Outputs list of tuples (path, takes) corresponding to the moves taking the maximum amount of pieces.
        (Cf. international checkers rules)
        - Player : 0 = white, 1 = black. 
        Note : turn % 2 = player.
        """
        if player == 1:
            opposite_moves = self.get_reverse().possible_moves(0)
            return [([self.size - 1 - i for i in move], [self.size - 1 - i for i in takes]) for move, takes in opposite_moves]
        
        path_queue = []
        possible_moves = [] 
        # List of moves that are physically possible on board

        for i in range(len(self.board)):
            if(self.board[i] > 0):
                path_queue.append(([i], []))
                # Path queue contains (path, pieces_taken) tuples.
        
        while len(path_queue):
            path, already_jumped = path_queue.pop()
            current_tile = path[-1]
            x, y = self.coordinates(current_tile)

            for x2 in range(0, self.side):
                if x2 != x:
                    # Try first diagonal move
                    y2 = y + x2 - x
                    if 0 <= y2 and y2 < self.side:
                        potential_tile = self.square_id((x2, y2))
                        piece_type = self.board[path[0]] 
                        # The piece stays at the beginning of the path until the move is done

                        valid, piece_taken = self.is_valid(current_tile, potential_tile, piece_type, already_jumped)
                        if valid:
                            new_path = path + [potential_tile]

                            if piece_taken is not None:
                                new_jumped_pieces = already_jumped + [piece_taken]
                                path_queue.append((new_path, new_jumped_pieces))
                                # If a piece is taken, longer move is possible

                            else:
                                new_jumped_pieces = already_jumped
                            possible_moves.append((new_path, new_jumped_pieces))

                    # Try second diagonal move
                    y2 = y + x - x2
                    if 0 <= y2 and y2 < self.side:
                        potential_tile = self.square_id((x2, y2))
                        piece_type = self.board[path[0]] 
                        # The piece stays at the beginning of the path until the move is done

                        valid, piece_taken = self.is_valid(current_tile, potential_tile, piece_type, already_jumped)
                        if valid:
                            new_path = path + [potential_tile]

                            if piece_taken is not None:
                                new_jumped_pieces = already_jumped + [piece_taken]
                                path_queue.append((new_path, new_jumped_pieces))
                                # If a piece is taken, longer move is possible

                            else:
                                new_jumped_pieces = already_jumped
                            possible_moves.append((new_path, new_jumped_pieces))
        
        max_pieces_taken = max([len(takes) for path, takes in possible_moves])
        valid_moves = [(p, takes) for p, takes in possible_moves if len(takes) == max_pieces_taken]

        return valid_moves