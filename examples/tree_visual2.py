import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

from graphviz import Digraph
import random

# Mock function to simulate state transitions for the sake of this example
def after_action_state(state, action):
    new_board = [row[:] for row in state[0]]  # Deep copy of the 3x3 board
    row, col = action
    new_board[row][col] = state[1]  # Place the mark at the chosen action
    return (new_board, next_mark(state[1]))

def next_mark(mark):
    return 'O' if mark == 'X' else 'X'

def check_game_status(board):
    # Check rows and columns for a win
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != '-':  # Row win
            return 1 if board[i][0] == 'O' else 2
        if board[0][i] == board[1][i] == board[2][i] != '-':  # Column win
            return 1 if board[0][i] == 'O' else 2

    # Check diagonals for a win
    if board[0][0] == board[1][1] == board[2][2] != '-':  # Top-left to bottom-right
        return 1 if board[0][0] == 'O' else 2
    if board[0][2] == board[1][1] == board[2][0] != '-':  # Top-right to bottom-left
        return 1 if board[0][2] == 'O' else 2

    # Check for a draw (if no empty cells left)
    for row in board:
        if '-' in row:
            return -1  # Continue playing

    return 0  # Draw

def evaluate_board(board):
    """ Evaluate board state at max depth to simulate potential outcomes. """
    # Evaluate rows and columns for possible wins
    for i in range(3):
        # Check rows
        if board[i].count('O') == 3:
            return O_REWARD
        if board[i].count('X') == 3:
            return X_REWARD
        # Check columns
        col = [board[0][i], board[1][i], board[2][i]]
        if col.count('O') == 3:
            return O_REWARD
        if col.count('X') == 3:
            return X_REWARD

    # Check diagonals for potential wins
    diagonal1 = [board[0][0], board[1][1], board[2][2]]
    diagonal2 = [board[0][2], board[1][1], board[2][0]]
    if diagonal1.count('O') == 3:
        return O_REWARD
    if diagonal1.count('X') == 3:
        return X_REWARD
    if diagonal2.count('O') == 3:
        return O_REWARD
    if diagonal2.count('X') == 3:
        return X_REWARD

    return DRAW_REWARD  # Neutral or draw state

# Constants for simplification
O_REWARD = 1   # Reward if 'O' wins
X_REWARD = -1  # Reward if 'X' wins
DRAW_REWARD = 0  # Reward for a draw
MINIMAX_DEPTH = 2

class MinimaxDraft:
    def __init__(self, mark):
        self.mark = mark

    def act(self, state):
        """ Always use Minimax to choose the best action. """
        return self.minimax(state)

    def minimax(self, state):
        """ Use the Minimax algorithm to choose the best action without recalling the environment. """
        
        # Store tree nodes for visualization
        self.tree = Digraph()
        self.node_counter = 0
        
        def minimax_recursive(state, mark, current_depth, is_maximizing, parent_id=None, action_taken=None):
            # Assign a unique node ID for visualization
            node_id = f"node{self.node_counter}"
            self.node_counter += 1
            
            # Check if the game has reached a terminal state (win, loss, or draw)
            status = check_game_status(state[0])
            if status != -1:  # Terminal state
                reward = O_REWARD if status == 1 else X_REWARD if status == 2 else DRAW_REWARD
                label = f"{'O wins' if status == 1 else 'X wins' if status == 2 else 'Draw'}\nReward: {reward}"
                self.tree.node(node_id, label, shape='box', color='blue' if is_maximizing else 'red')
                if parent_id:
                    self.tree.edge(parent_id, node_id, label=str(action_taken))
                return reward

            # If depth limit reached, evaluate the board
            if current_depth >= MINIMAX_DEPTH:
                reward = evaluate_board(state[0])
                label = f"Max Depth\nReward: {reward}"
                self.tree.node(node_id, label, shape='box', style='dashed')
                if parent_id:
                    self.tree.edge(parent_id, node_id, label=str(action_taken))
                return reward

            # Generate available actions based on the current board state
            ava_actions = [(i, j) for i in range(3) for j in range(3) if state[0][i][j] == '-']

            # Continue with the Minimax algorithm, exploring future states
            player_label = f"Player {mark}"
            if is_maximizing:
                max_eval = -float('inf')
                best_actions = []  # List to track best actions
                self.tree.node(node_id, f"Maximizing - {player_label} - Depth {current_depth}", color='blue')
                if parent_id:
                    self.tree.edge(parent_id, node_id, label=str(action_taken))
                
                for action in ava_actions:
                    new_state = after_action_state(state, action)
                    eval = minimax_recursive(new_state, next_mark(mark), current_depth + 1, False, node_id, action)
                    
                    if eval > max_eval:
                        max_eval = eval
                        best_actions = [action]  # Reset best actions list
                    elif eval == max_eval:
                        best_actions.append(action)  # Add to best actions list

                chosen_action = random.choice(best_actions) if current_depth == 0 else None
                return chosen_action if current_depth == 0 else max_eval

            else:
                min_eval = float('inf')
                best_actions = []  # List to track best actions
                self.tree.node(node_id, f"Minimizing - {player_label} - Depth {current_depth}", color='red')
                if parent_id:
                    self.tree.edge(parent_id, node_id, label=str(action_taken))

                for action in ava_actions:
                    new_state = after_action_state(state, action)
                    eval = minimax_recursive(new_state, next_mark(mark), current_depth + 1, True, node_id, action)
                    
                    if eval < min_eval:
                        min_eval = eval
                        best_actions = [action]  # Reset best actions list
                    elif eval == min_eval:
                        best_actions.append(action)  # Add to best actions list

                chosen_action = random.choice(best_actions) if current_depth == 0 else None
                return chosen_action if current_depth == 0 else min_eval

        # Start visualization from the root
        best_action = minimax_recursive(state, self.mark, 0, self.mark == 'O')
        self.tree.render("minimax_tree", format='png', directory=".", cleanup=True)  # Save the graph as an image
        return best_action

# Example to run the MinimaxDraft and generate a decision tree
if __name__ == '__main__':
    # Intermediate game state
    initial_state = ([
        ['O', 'X', 'O'],
        ['-', 'O', '-'],
        ['X', '-', 'X']
    ], 'O')  # 'O' to play next
    agent = MinimaxDraft('O')
    best_move = agent.act(initial_state)

    print(f"Best move for O: {best_move}")
    print("Minimax tree generated as 'minimax_tree.png'")
