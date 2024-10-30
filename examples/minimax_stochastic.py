#!/usr/bin/env python
import os
import sys
import random
import time
import logging
import json
from collections import defaultdict
from itertools import product
from multiprocessing import Pool
from tempfile import NamedTemporaryFile

import pandas as pd
import click
from tqdm import tqdm as _tqdm
tqdm = _tqdm

from gym_tictactoe.env import TicTacToeEnv, set_log_level_by, agent_by_mark,\
    next_mark, check_game_status, after_action_state, O_REWARD, X_REWARD
from human_agent import HumanAgent

# Depth for Minimax algorithm
MINIMAX_DEPTH = 1

# Rewards for winning or losing
O_REWARD = 1   # Reward if 'O' wins
X_REWARD = -1  # Reward if 'X' wins
DRAW_REWARD = 0  # Reward for a draw
MAX_DEPTH_REWARD = 0 # Reward when reaching max depth

class MinimaxStochastic(object):
    def __init__(self, mark):
        self.mark = mark

    def act(self, state, ava_actions):
        """ Always use Minimax to choose the best action. """
        return self.minimax(state, ava_actions)


    def minimax(self, state, ava_actions):
        """ Use the Minimax algorithm to choose the best action without recalling the environment. """

        def minimax_recursive(state, mark, current_depth, is_maximizing):
            # Check if the game has reached a terminal state (win, loss, or draw)
            status = check_game_status(state[0])
            if status != -1:  # Terminal state
                if status == 1:
                    return O_REWARD  # 'O' wins
                elif status == 2:
                    return X_REWARD  # 'X' wins
                else:
                    return DRAW_REWARD  # Draw

            # If depth limit reached, return the default value for non-terminal states
            if current_depth >= MINIMAX_DEPTH:
                return MAX_DEPTH_REWARD 
            
            # Get available actions from the current state
            board, _ = state
            ava_actions = [i for i in range(len(board)) if board[i] == 0]
            

            # Continue with the Minimax algorithm, exploring future states
            if is_maximizing:
                max_eval = -float('inf')
                best_actions = []  # List to track best actions
                for action in ava_actions:
                    new_state = after_action_state(state, action)
                    eval = minimax_recursive(new_state, next_mark(mark), current_depth + 1, False)
                    
                    if eval > max_eval:
                        max_eval = eval
                        best_actions = [action]  # Reset best actions list
                    elif eval == max_eval:
                        best_actions.append(action)  # Add to best actions list

                return random.choice(best_actions) if current_depth == 0 else max_eval

            else:
                min_eval = float('inf')
                best_actions = []  # List to track best actions
                for action in ava_actions:
                    new_state = after_action_state(state, action)
                    eval = minimax_recursive(new_state, next_mark(mark), current_depth + 1, True)
                    
                    if eval < min_eval:
                        min_eval = eval
                        best_actions = [action]  # Reset best actions list
                    elif eval == min_eval:
                        best_actions.append(action)  # Add to best actions list

                return random.choice(best_actions) if current_depth == 0 else min_eval

        # Call the recursive function within minimax
        best_action = minimax_recursive(state, self.mark, 0, self.mark == 'O')
        return best_action


@click.command(help="Play with human.")
@click.option('-l', '--load-file', type=str, default=None,
              show_default=True, help="Path to a file to load a learned model.")
@click.option('-n', '--show-number', is_flag=True, default=False,
              show_default=True, help="Show location number when play.")
def play(load_file, show_number):
    _play(load_file, HumanAgent('O'), show_number)

def _play(load_file, vs_agent, show_number):
    env = TicTacToeEnv(show_number=show_number)
    td_agent = MinimaxStochastic('X')
    start_mark = 'O'
    agents = [vs_agent, td_agent]

    while True:
        env.set_start_mark(start_mark)
        state = env.reset()
        _, mark = state
        done = False

        if mark == 'O':
            env.render(mode='human')

        while not done:
            agent = agent_by_mark(agents, mark)
            human = isinstance(agent, HumanAgent)

            env.show_turn(True, mark)
            ava_actions = env.available_actions()
            action = agent.act(state, ava_actions) if not human else agent.act(ava_actions)
            if human and action is None:
                sys.exit()

            state, reward, done, info = env.step(action)
            env.render(mode='human')

            if done:
                env.show_result(True, mark, reward)
                break
            else:
                _, mark = state

        start_mark = next_mark(start_mark)

if __name__ == '__main__':
    play()