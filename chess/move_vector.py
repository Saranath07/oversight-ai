import gym_chess  # <-- This must come before gym.make
import gym
import chess

env = gym.make('ChessAlphaZero-v0')
env.reset()

move = chess.Move.from_uci('e2e4')
print(env.encode(move))  # Should print an integer, e.g., 877
