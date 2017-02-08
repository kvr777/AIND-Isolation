"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!

    #General part
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    #To select heuristic
    selected_score = 2

    # weighted improved_score as it described in lecture
    if selected_score==1:
        own_moves = len(game.get_legal_moves(player))
        opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
        return float(own_moves - 2*opp_moves)

    # additional heuristic with
    elif selected_score==2:
        own_moves = len(game.get_legal_moves(player))
        opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
        game_deep_ratio = float(len(game.get_blank_spaces())/(game.width*game.height*1.0))
        center_w=game.width/2.0
        center_h=game.height/2.0
        current_loc= game.get_player_location(player)
        coeff= 1 /(game_deep_ratio**2)
        proximity_to_center = abs(current_loc[0]-center_h) + abs(current_loc[1]-center_w)
        return float(own_moves - 3*opp_moves*coeff - 3*proximity_to_center/coeff)

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left
        move = (-1, -1)
        # Return (-1, -1) for no legal moves
        if len(legal_moves) == 0:
            return move

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            # initialize time margin in ms to o leave some time for the function to return
            time_margin = 50

            if self.iterative:
                depth=1
            else: depth = self.search_depth

            while self.time_left()>time_margin:
                if self.method == 'minimax':
                    _, move = self.minimax(game, depth)
                else:
                    if self.method == 'alphabeta':
                        _, move = self.alphabeta(game, depth)

                if not self.iterative:
                    return move
                depth += 1

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return move

        # Return the best move from the last completed search iteration
        return move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        #get legal moves

        legal_moves = game.get_legal_moves()
        best_move = (-1, -1)

        #Return (-1, -1) for no legal moves
        if len(legal_moves)==0 or depth==0:
            return self.score(game,self), best_move

        # Iterative minimax search according to AIMA algorithm
        if maximizing_player:
            best_value = float('-inf')
            for move in legal_moves:
                    v,_ = self.minimax(game.forecast_move(move), depth - 1, False)
                    if v > best_value:
                        best_value = v
                        best_move = move
            return best_value, best_move
        else:
            best_value = float('inf')
            for move in legal_moves:
                v, _ = self.minimax(game.forecast_move(move), depth - 1, True)
                if v < best_value:
                    best_value = v
                    best_move = move

            return best_value, best_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        #get legal moves

        legal_moves = game.get_legal_moves()

        best_move = (-1,-1)



        #Return (-1, -1) for no legal moves
        if len(legal_moves)==0 or depth==0:
            return self.score(game,self), best_move

        # Iterative alphabeta search according to AIMA algorithm
        if maximizing_player:
            best_value = float('-inf')
            for move in legal_moves:
                v,_ = self.alphabeta(game.forecast_move(move), depth - 1,alpha, beta, False)
                #find the nodes we can cut-off
                if v >= beta:
                    return v, move
                #otherwise update best values
                if v>= best_value:
                    best_value=v
                    best_move = move
                    alpha = max(alpha, best_value)
            return best_value, best_move

        else:
            best_value = float('inf')
            for move in legal_moves:
                v, _ = self.alphabeta(game.forecast_move(move), depth - 1, alpha, beta, True)
                # find the nodes we can cut-off
                if v <= alpha:
                    return v, move
                # otherwise update best values
                if v < best_value:
                    best_value = v
                    best_move = move
                    beta = min(beta, best_value)
            return best_value, best_move