# Description

This project (Reinforcement Learning, Ecole Polytechnique, 2024) aims to produce and compare several agents able to play the checkers game. We developed a suitable environment to play the checkers. On this environment, a Q-learning agent, a Min-max agent, a tree-agent and a random agent can play against each other. We compared how they would perform in order to build a ranking between them and establish the best approach if there is one.

# Files

The Board class (board.py) represents a board and gives its allowed moves; board_metrics.py implements a reward corresponding to a board state.
The GameRunner class (game_runner.py) contains the logic related to the game and allows two agents to play against each other.

All agents are derived from the generic Agent class (abstract_agent.py):
- tree_agent.py implements an agent taking its decisions according to a decision tree
- minimax_agent.py represents a simple agent following the minimax algorithm
- deepQ_agent.py represents a Q-learning agent
- naive_deep_qlearning_agent.py is a "naive" Q-learning agent (cf. report for details), whose network is characterized by a large output layer. naive_deep_qlearning.py and naive_deep_qlearning_train.py perform the training of this agent. The agent can either be rewarded after each move, or at the end of a game. See the report for more details.
- deepQ_agent.py is the improved version of this Q-learning agent, with a smaller and more convenient output layer. deepQ_utils.py and deepQ_train.py are used for its training.

Finally, test.py tests one desired agent against another and results are presented in report.pdf
