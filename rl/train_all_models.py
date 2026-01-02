#!/usr/bin/env python3
"""
Train all model variants for comparison.

This script trains multiple model architectures:
1. Basic DQN (longer training - 100k episodes)
2. DQN with reward shaping (50k episodes)
3. CNN DQN with reward shaping (50k episodes)

Usage:
    uv run train_all_models.py
"""

import argparse
import os
import time
from datetime import datetime
from typing import Dict

import numpy as np
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from game_env import Game
from dqn_agent import DQNAgent
from dqn_cnn_agent import DQNAgent_CNN, encode_board_onehot, shape_reward


def train_model(
    agent,
    model_name: str,
    episodes: int,
    seed: int,
    use_cnn: bool = False,
    use_reward_shaping: bool = False,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = None,
    eval_freq: int = 2000,
    eval_episodes: int = 50
):
    """Train a model and save checkpoints."""
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    writer = None
    if HAS_TENSORBOARD and log_dir:
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard: {log_dir}")
    
    best_avg_score = 0
    
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Episodes: {episodes}, CNN: {use_cnn}, Reward Shaping: {use_reward_shaping}")
    print(f"Device: {agent.device}")
    print(f"{'='*60}\n")
    
    pbar = tqdm(range(episodes), desc=model_name)
    for episode in pbar:
        game = Game(seed=seed + episode)
        
        if use_cnn:
            state = encode_board_onehot(np.array(game.board))
        else:
            state = np.array(game.get_state(), dtype=np.float32)
        
        total_reward = 0
        steps = 0
        
        while not game.is_done() and steps < 10000:
            legal_actions = game.legal_actions()
            action = agent.select_action(state, legal_actions, training=True)
            
            result = game.step(action)
            
            if use_cnn:
                next_state = encode_board_onehot(np.array(game.board))
            else:
                next_state = np.array(game.get_state(), dtype=np.float32)
            
            # Reward shaping
            if use_reward_shaping:
                reward = shape_reward(game, result, action)
            else:
                reward = result['reward']
                if not result['changed']:
                    reward = -10
            
            agent.store_experience(state, action, reward, next_state, result['done'])
            agent.update()
            
            state = next_state
            total_reward += result['reward']
            steps += 1
        
        pbar.set_postfix({
            'score': game.score,
            'max': game.max_tile(),
            'ε': f"{agent.epsilon:.2f}"
        })
        
        if writer:
            writer.add_scalar(f'{model_name}/score', game.score, episode)
            writer.add_scalar(f'{model_name}/max_tile', game.max_tile(), episode)
            writer.add_scalar(f'{model_name}/epsilon', agent.epsilon, episode)
        
        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_scores = []
            eval_max_tiles = []
            
            for i in range(eval_episodes):
                eval_game = Game(seed=seed + 1000000 + i)
                if use_cnn:
                    eval_state = encode_board_onehot(np.array(eval_game.board))
                else:
                    eval_state = np.array(eval_game.get_state(), dtype=np.float32)
                
                while not eval_game.is_done():
                    legal = eval_game.legal_actions()
                    a = agent.select_action(eval_state, legal, training=False)
                    eval_game.step(a)
                    if use_cnn:
                        eval_state = encode_board_onehot(np.array(eval_game.board))
                    else:
                        eval_state = np.array(eval_game.get_state(), dtype=np.float32)
                
                eval_scores.append(eval_game.score)
                eval_max_tiles.append(eval_game.max_tile())
            
            avg_score = np.mean(eval_scores)
            avg_max = np.mean(eval_max_tiles)
            
            print(f"\n[{model_name} @ {episode+1}] Avg: {avg_score:.0f}, Max Tile: {avg_max:.0f}")
            
            if writer:
                writer.add_scalar(f'{model_name}/eval_avg_score', avg_score, episode)
                writer.add_scalar(f'{model_name}/eval_avg_max_tile', avg_max, episode)
            
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                agent.save(os.path.join(checkpoint_dir, f'{model_name}_best.pt'))
                print(f"  New best! Saved {model_name}_best.pt")
    
    # Final save
    agent.save(os.path.join(checkpoint_dir, f'{model_name}_final.pt'))
    
    if writer:
        writer.close()
    
    print(f"\n{model_name} complete! Best avg: {best_avg_score:.0f}")
    return best_avg_score


def main():
    parser = argparse.ArgumentParser(description='Train all model variants')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--skip-basic', action='store_true', help='Skip basic DQN')
    parser.add_argument('--skip-shaped', action='store_true', help='Skip reward-shaped DQN')
    parser.add_argument('--skip-cnn', action='store_true', help='Skip CNN DQN')
    parser.add_argument('--basic-episodes', type=int, default=100000, help='Episodes for basic DQN')
    parser.add_argument('--shaped-episodes', type=int, default=50000, help='Episodes for shaped DQN')
    parser.add_argument('--cnn-episodes', type=int, default=50000, help='Episodes for CNN DQN')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = 'checkpoints'
    
    results = {}
    
    # Model 1: Basic DQN (longer training)
    if not args.skip_basic:
        agent = DQNAgent(
            epsilon_decay=200000,  # Slower decay for more exploration
            batch_size=256
        )
        results['dqn_basic_100k'] = train_model(
            agent=agent,
            model_name='dqn_basic_100k',
            episodes=args.basic_episodes,
            seed=args.seed,
            use_cnn=False,
            use_reward_shaping=False,
            checkpoint_dir=checkpoint_dir,
            log_dir=f'runs/dqn_basic_100k_{timestamp}',
            eval_freq=5000
        )
    
    # Model 2: DQN with reward shaping
    if not args.skip_shaped:
        agent = DQNAgent(
            epsilon_decay=100000,
            batch_size=256
        )
        results['dqn_shaped'] = train_model(
            agent=agent,
            model_name='dqn_shaped',
            episodes=args.shaped_episodes,
            seed=args.seed + 100000,
            use_cnn=False,
            use_reward_shaping=True,
            checkpoint_dir=checkpoint_dir,
            log_dir=f'runs/dqn_shaped_{timestamp}',
            eval_freq=2500
        )
    
    # Model 3: CNN DQN with reward shaping
    if not args.skip_cnn:
        agent = DQNAgent_CNN(
            epsilon_decay=100000,
            batch_size=128,  # Smaller batch for CNN memory
            use_reward_shaping=True
        )
        results['dqn_cnn'] = train_model(
            agent=agent,
            model_name='dqn_cnn',
            episodes=args.cnn_episodes,
            seed=args.seed + 200000,
            use_cnn=True,
            use_reward_shaping=True,
            checkpoint_dir=checkpoint_dir,
            log_dir=f'runs/dqn_cnn_{timestamp}',
            eval_freq=2500
        )
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    for name, score in results.items():
        print(f"  {name}: {score:.0f} avg score")
    print("="*60)


if __name__ == "__main__":
    main()
