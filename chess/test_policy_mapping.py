import gym_chess  # <-- This must come before gym.make
import gym
import chess
import numpy as np

# Initialize the environment
env = gym.make('ChessAlphaZero-v0')
env.reset()

# Test various moves to understand the mapping
test_moves = [
    'e2e4',  # Pawn move
    'g1f3',  # Knight move
    'f1c4',  # Bishop move
    'e1g1',  # Castling
    'e7e8q', # Pawn promotion to queen
    'e7e8n', # Pawn promotion to knight (underpromotion)
    'a2a3',  # Another pawn move
    'h1h8',  # Rook move (long distance)
]

print("Testing UCI to Policy Index mapping:")
print("=" * 50)

for uci_move in test_moves:
    try:
        move = chess.Move.from_uci(uci_move)
        policy_index = env.encode(move)
        print(f"UCI: {uci_move:6} -> Policy Index: {policy_index:4}")
    except Exception as e:
        print(f"UCI: {uci_move:6} -> Error: {e}")

# Test the reverse mapping (if available)
print("\nTesting Policy Index to UCI mapping:")
print("=" * 50)

# Test if we can decode back
test_indices = [877, 0, 100, 500, 1000, 2000, 3000, 4000]
for idx in test_indices:
    try:
        # Check if decode method exists
        if hasattr(env, 'decode'):
            move = env.decode(idx)
            print(f"Policy Index: {idx:4} -> UCI: {move}")
        else:
            print(f"Policy Index: {idx:4} -> Decode method not available")
            break
    except Exception as e:
        print(f"Policy Index: {idx:4} -> Error: {e}")

# Get the action space size
print(f"\nAction space size: {env.action_space.n}")
print(f"Expected size (AlphaZero): 4672")