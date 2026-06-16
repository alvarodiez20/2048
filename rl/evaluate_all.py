#!/usr/bin/env python3
"""
Evaluate all existing model checkpoints to measure their performance.
Reports: avg score, max tile distribution, % reaching 512/1024/2048.

This script handles both old (pre-Dueling) and new architectures by
introspecting checkpoint state dicts.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Dict, List

# Add rl directory to path
sys.path.insert(0, os.path.dirname(__file__))
from game_env import Game


# ============================================================
# Old DQN architecture (matches pre-upgrade checkpoints)
# ============================================================
class OldDQN(nn.Module):
    """Original MLP DQN: 16 -> 256 -> 256 -> 128 -> 4"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class OldDQN_CNN(nn.Module):
    """Original CNN DQN: 2 conv layers + FC"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 64, kernel_size=2, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=2, padding=0)
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def encode_board_onehot(board):
    """Encode board as one-hot tensor: (16, 4, 4)"""
    result = np.zeros((16, 4, 4), dtype=np.float32)
    for i in range(16):
        row, col = i // 4, i % 4
        val = board[i]
        if val == 0:
            result[0, row, col] = 1.0
        else:
            channel = int(np.log2(val))
            if channel < 16:
                result[channel, row, col] = 1.0
    return result


def load_model(checkpoint_path: str, is_cnn: bool = False):
    """Load a model checkpoint, auto-detecting old vs new architecture."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['policy_net']

    if is_cnn:
        # Check for old architecture keys
        if 'conv1.weight' in state_dict and 'bn1.weight' not in state_dict:
            model = OldDQN_CNN()
        else:
            from dqn_cnn_agent import DQN_CNN
            model = DQN_CNN()
    else:
        # Check for old architecture keys (has fc1..fc4 but not shared1)
        if 'fc1.weight' in state_dict and 'shared1.weight' not in state_dict:
            model = OldDQN()
        else:
            from dqn_agent import DuelingDQN
            model = DuelingDQN()

    model.load_state_dict(state_dict)
    model.eval()
    return model, is_cnn


def evaluate_model(model, is_cnn: bool, num_episodes: int = 200, seed: int = 42):
    """Run evaluation episodes and return stats."""
    scores = []
    max_tiles = []
    steps_list = []

    for i in range(num_episodes):
        game = Game(seed=seed + i)
        state = np.array(game.get_state(), dtype=np.float32)
        steps = 0

        while not game.is_done() and steps < 10000:
            legal_actions = game.legal_actions()
            valid_actions = [j for j, legal in enumerate(legal_actions) if legal]

            if not valid_actions:
                break

            with torch.no_grad():
                if is_cnn:
                    # Get raw board for CNN encoding
                    board = game.board if isinstance(game.board, list) else list(game.board)
                    encoded = encode_board_onehot(board)
                    state_tensor = torch.FloatTensor(encoded).unsqueeze(0)
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)

                q_values = model(state_tensor).squeeze()

                # Mask invalid actions
                for j in range(4):
                    if j not in valid_actions:
                        q_values[j] = float('-inf')

                action = q_values.argmax().item()

            result = game.step(action)
            state = np.array(game.get_state(), dtype=np.float32)
            steps += 1

        scores.append(game.score)
        max_tiles.append(game.max_tile())
        steps_list.append(steps)

    return {
        'avg_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': np.max(scores),
        'median_score': np.median(scores),
        'avg_steps': np.mean(steps_list),
        'tile_dist': _tile_distribution(max_tiles, num_episodes),
        'pct_512': sum(1 for t in max_tiles if t >= 512) / num_episodes * 100,
        'pct_1024': sum(1 for t in max_tiles if t >= 1024) / num_episodes * 100,
        'pct_2048': sum(1 for t in max_tiles if t >= 2048) / num_episodes * 100,
    }


def evaluate_random(num_episodes: int = 200, seed: int = 42):
    """Random baseline."""
    random.seed(seed)
    scores = []
    max_tiles = []

    for i in range(num_episodes):
        game = Game(seed=seed + i)
        steps = 0
        while not game.is_done() and steps < 10000:
            legal = game.legal_actions()
            valid = [j for j, l in enumerate(legal) if l]
            if not valid:
                break
            game.step(random.choice(valid))
            steps += 1
        scores.append(game.score)
        max_tiles.append(game.max_tile())

    return {
        'avg_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': np.max(scores),
        'median_score': np.median(scores),
        'tile_dist': _tile_distribution(max_tiles, num_episodes),
        'pct_512': sum(1 for t in max_tiles if t >= 512) / num_episodes * 100,
        'pct_1024': sum(1 for t in max_tiles if t >= 1024) / num_episodes * 100,
        'pct_2048': sum(1 for t in max_tiles if t >= 2048) / num_episodes * 100,
    }


def _tile_distribution(max_tiles, num_episodes):
    dist = {}
    for t in max_tiles:
        dist[t] = dist.get(t, 0) + 1
    return {k: f"{v} ({v/num_episodes*100:.1f}%)" for k, v in sorted(dist.items())}


def main():
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    num_episodes = 200

    models_to_eval = [
        ('best_model.pt', 'Best MLP (10k)', False),
        ('dqn_basic_100k_best.pt', 'Basic MLP (100k)', False),
        ('dqn_shaped_best.pt', 'Shaped MLP', False),
        ('dqn_cnn_best.pt', 'CNN DQN', True),
    ]

    print("=" * 70)
    print(f"EVALUATING ALL MODELS ({num_episodes} episodes each)")
    print("=" * 70)

    # Random baseline
    print("\n--- Random Baseline ---")
    rand_stats = evaluate_random(num_episodes)
    print(f"  Avg Score: {rand_stats['avg_score']:.0f} ± {rand_stats['std_score']:.0f}")
    print(f"  Max Score: {rand_stats['max_score']}")
    print(f"  >=512: {rand_stats['pct_512']:.1f}%  >=1024: {rand_stats['pct_1024']:.1f}%  >=2048: {rand_stats['pct_2048']:.1f}%")
    print(f"  Tile dist: {rand_stats['tile_dist']}")

    # Evaluate each model
    for checkpoint_name, display_name, is_cnn in models_to_eval:
        path = os.path.join(checkpoint_dir, checkpoint_name)
        if not os.path.exists(path):
            print(f"\n--- {display_name} --- SKIPPED (not found: {checkpoint_name})")
            continue

        print(f"\n--- {display_name} ({checkpoint_name}) ---")
        try:
            model, cnn = load_model(path, is_cnn)
            stats = evaluate_model(model, cnn, num_episodes)
            print(f"  Avg Score: {stats['avg_score']:.0f} ± {stats['std_score']:.0f}")
            print(f"  Max Score: {stats['max_score']}")
            print(f"  Median:   {stats['median_score']:.0f}")
            print(f"  Avg Steps: {stats['avg_steps']:.0f}")
            print(f"  >=512: {stats['pct_512']:.1f}%  >=1024: {stats['pct_1024']:.1f}%  >=2048: {stats['pct_2048']:.1f}%")
            print(f"  Tile dist: {stats['tile_dist']}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
