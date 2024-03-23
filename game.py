import random
import copy
import numpy as np
import time

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        # Drop phase?
        drop_phase = sum(p != ' ' for row in state for p in row) < 8

        # best move from max_val
        value, best_state = self.max_value(state, 0)

        # curretn state and best state diff
        diff_state = np.array(state) != np.array(best_state)
        changes = np.where(diff_state)

        # coordinates
        coord = list(zip(changes[0], changes[1]))

        if drop_phase:
            # first tuple where a piece is placed
            for row, col in coord:

                if best_state[row][col] != ' ':
                    return [(row, col)]
        else:
            # tuple with  source and destination
            orig = next((row, col) for row, col in coord if state[row][col] != ' ')
            dest = next((row, col) for row, col in coord if best_state[row][col] != ' ')

            return [dest, orig]

        # no move found (shouldn't happen tho)
        return []
    
    def succ(self, state, piece):
        
        self.game_value(state)
        succ = []
        drop_phase = sum(i.count('b') for i in state) < 4 or sum(i.count('r') for i in state) < 4

        def move(state, i, j, di, dj):

            if 0 <= i + di < len(state) and 0 <= j + dj < len(state) and state[i + di][j + dj] == ' ':

                new_state = copy.deepcopy(state)
                new_state[i][j], new_state[i + di][j + dj] = new_state[i + di][j + dj], new_state[i][j]

                return new_state

        if not drop_phase:
            for row in range(len(state)):
                for col in range(len(state)):

                    if state[row][col] == piece:
                        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

                        for di, dj in directions:

                            new_state = move(state, row, col, di, dj)

                            if new_state:
                                succ.append(new_state)
        else:
            for row in range(len(state)):
                for col in range(len(state)):

                    if state[row][col] == ' ':
                        new_state = copy.deepcopy(state)
                        new_state[row][col] = piece
                        succ.append(new_state)

        return list(filter(None, succ))

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # TODO: check \ diagonal wins
        for row in range(2):
            for col in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col+1] == state[row+2][col+2] == state[row+3][col+3]:
                    return 1 if state[row][col]==self.my_piece else -1
                
        # TODO: check / diagonal wins
        for row in range(2):
            for col in range(3,5):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col-1] == state[row+2][col-2] == state[row+3][col-3]:
                    return 1 if state[row][col]==self.my_piece else -1
                
        # TODO: check box wins
        for row in range(4):
            for col in range(4):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col] == state[row][col+1] == state[row+1][col+1]:
                    return 1 if state[row][col]==self.my_piece else -1

        return 0 # no winner yet
    
    def heuristic_game_value(self, state, piece):
        
        my_piece = piece
        ai_piece = 'r' if piece == 'b' else 'b'
        
        def count_connected(row_range, col_range):
            my_count, ai_count = 0, 0
            
            for row in row_range:
                for col in col_range:
                    if state[row][col] == my_piece:
                        my_count += 1
                        
                    elif state[row][col] == ai_piece:
                        ai_count += 1

            return my_count, ai_count
        
        my_max, ai_max = 0, 0
        
        # Horizontal
        for i in range(len(state)):
            my_count, ai_count = count_connected([i], range(len(state)))
            my_max = max(my_max, my_count)
            ai_max = max(ai_max, ai_count)
            
        # Vertical
        for i in range(len(state)):
            my_count, ai_count = count_connected(range(len(state)), [i])
            my_max = max(my_max, my_count)
            ai_max = max(ai_max, ai_count)
            
        # Diagonal /
        for row in range(3, 5):
            my_count, ai_count = count_connected(range(row, row - 4, -1), range(2))
            my_max = max(my_max, my_count)
            ai_max = max(ai_max, ai_count)
            
        # Diagonal \
        for row in range(2):
            my_count, ai_count = count_connected(range(row, row + 4), range(2))
            my_max = max(my_max, my_count)
            ai_max = max(ai_max, ai_count)
            
        # 2x2 Box
        for row in range(4):
            for i in range(4):
                my_count, ai_count = count_connected(range(row, row + 2), range(i, i + 2))
                my_max = max(my_max, my_count)
                ai_max = max(ai_max, ai_count)
                
        if my_max == ai_max:
            return 0, state
        
        if my_max >= ai_max:
            return my_max / 6, state
        
        return -ai_max / 6, state

        
    def max_value(self, state, depth):

        best_state = state

        if self.game_value(state) != 0:
            return self.game_value(state),state

        if depth >= 2:
            return self.heuristic_game_value(state,self.my_piece)

        else:
            best_val = float('-Infinity')

            for succ in self.succ(state, self.my_piece):
                value = self.min_value(succ,depth+1)

                if value[0] > best_val:
                    best_val = value[0]
                    best_state = succ

        return best_val, best_state

    def min_value(self, state,depth):

        best_state = state

        if self.game_value(state) != 0:
            return self.game_value(state),state
        
        if depth >= 2:
            return self.heuristic_game_value(state, self.opp)

        else:
            best_val = float('Infinity')

            for succ in self.succ(state, self.opp):
                value = self.max_value(succ, depth + 1)

                if value[0] < best_val:
                    best_val = value[0]
                    best_state = succ

        return best_val, best_state

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            start = time.time()
            move = ai.make_move(ai.board)
            end = time.time()
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
            print("Time:", end-start)
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            start = time.time()
            move = ai.make_move(ai.board)
            end = time.time()
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
            print("Time:", end-start)
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()