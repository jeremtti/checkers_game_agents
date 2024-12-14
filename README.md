# Description

This project (Reinforcement Learning, Ecole Polytechnique, 2024) aims to produce and compare several agents able to play the checkers game. We developed a suitable environment to play the checkers. On this environment, a Q-learning agent, a Min-max agent, a tree-agent and a random agent can play against each other. We compared how they would perform in order to build a ranking between them and establish the best approach if there is one.

# Files

The Board class (board.py) represents a board and gives its allowed moves.
The GameRunner class (game_runner.py) contains the logic related to the game and allows two agents to play against each other.

All agents are derived from the generic Agent class (abstract_agent.py):
