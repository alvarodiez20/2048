"""
Training script for 2048 DQN Agent

Usage:
    python train.py --episodes 10000 --seed 42
    python train.py --episodes 50000 --log-dir runs/experiment1
"""

import argparse
import os
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("TensorBoard not available. Install with: pip install tensorboard")

from game_env import Game
from dqn_agent import DQNAgent


def train_episode(game: Game, agent: DQNAgent, max_steps: int = 10000) -> Dict:
    """
    Train for one episode.
    
    Returns:
        Dict with episode statistics
    """
    state = np.array(game.get_state(), dtype=np.float32)
    total_reward = 0
    total_loss = 0
    loss_count = 0
    steps = 0
    invalid_moves = 0
    
    while not game.is_done() and steps < max_steps:
        legal_actions = game.legal_actions()
        
        # Select action
        action = agent.select_action(state, legal_actions, training=True)
        
        # Take action
        result = game.step(action)
        next_state = np.array(game.get_state(), dtype=np.float32)
        
        # Reward shaping
        reward = result['reward']
        
        # Penalty for invalid moves (shouldn't happen with legal_actions mask)
        if not result['changed']:
            reward = -10
            invalid_moves += 1
        
        # Bonus for higher tiles (encourage progress)
        max_tile = game.max_tile()
        if max_tile >= 2048:
            reward += 100
        elif max_tile >= 1024:
            reward += 20
        elif max_tile >= 512:
            reward += 5
        
        # Store experience
        agent.store_experience(state, action, reward, next_state, result['done'])
        
        # Update agent
        loss = agent.update()
        if loss is not None:
            total_loss += loss
            loss_count += 1
        
        state = next_state
        total_reward += result['reward']
        steps += 1
    
    return {
        'score': game.score,
        'max_tile': game.max_tile(),
        'steps': steps,
        'total_reward': total_reward,
        'avg_loss': total_loss / max(1, loss_count),
        'epsilon': agent.epsilon,
        'invalid_moves': invalid_moves
    }


def evaluate(game_class, agent: DQNAgent, num_episodes: int = 100, seed: int = 0) -> Dict:
    """
    Evaluate the agent without exploration.
    
    Returns:
        Dict with evaluation statistics
    """
    scores = []
    max_tiles = []
    steps_list = []
    
    for i in range(num_episodes):
        game = game_class(seed=seed + i)
        state = np.array(game.get_state(), dtype=np.float32)
        steps = 0
        
        while not game.is_done() and steps < 10000:
            legal_actions = game.legal_actions()
            action = agent.select_action(state, legal_actions, training=False)
            result = game.step(action)
            state = np.array(game.get_state(), dtype=np.float32)
            steps += 1
        
        scores.append(game.score)
        max_tiles.append(game.max_tile())
        steps_list.append(steps)
    
    # Tile distribution
    tile_counts = {}
    for tile in max_tiles:
        tile_counts[tile] = tile_counts.get(tile, 0) + 1
    
    return {
        'avg_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': np.max(scores),
        'min_score': np.min(scores),
        'avg_max_tile': np.mean(max_tiles),
        'avg_steps': np.mean(steps_list),
        'tile_distribution': tile_counts,
        'scores': scores,
        'max_tiles': max_tiles
    }


def main():
    parser = argparse.ArgumentParser(description='Train DQN agent for 2048')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Initial epsilon')
    parser.add_argument('--epsilon-end', type=float, default=0.01, help='Final epsilon')
    parser.add_argument('--epsilon-decay', type=int, default=100000, help='Epsilon decay steps')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--buffer-size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--target-update', type=int, default=1000, help='Target network update frequency')
    parser.add_argument('--eval-freq', type=int, default=1000, help='Evaluation frequency (episodes)')
    parser.add_argument('--eval-episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--save-freq', type=int, default=1000, help='Checkpoint save frequency')
    parser.add_argument('--log-dir', type=str, default=None, help='TensorBoard log directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--max-steps', type=int, default=10000, help='Max steps per episode')
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # TensorBoard
    writer = None
    if HAS_TENSORBOARD:
        log_dir = args.log_dir or f"runs/dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs: {log_dir}")
    
    # Create agent
    agent = DQNAgent(
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        target_update_freq=args.target_update
    )
    print(f"Agent device: {agent.device}")
    
    # Training loop
    best_avg_score = 0
    episode_scores = []
    episode_max_tiles = []
    
    print(f"\nStarting training for {args.episodes} episodes...")
    start_time = time.time()
    
    pbar = tqdm(range(args.episodes), desc="Training")
    for episode in pbar:
        # Create game for this episode
        game = Game(seed=args.seed + episode)
        
        # Train episode
        stats = train_episode(game, agent, max_steps=args.max_steps)
        
        episode_scores.append(stats['score'])
        episode_max_tiles.append(stats['max_tile'])
        
        # Update progress bar
        recent_avg = np.mean(episode_scores[-100:]) if episode_scores else 0
        pbar.set_postfix({
            'score': stats['score'],
            'avg100': f"{recent_avg:.0f}",
            'max_tile': stats['max_tile'],
            'ε': f"{stats['epsilon']:.3f}"
        })
        
        # TensorBoard logging
        if writer:
            writer.add_scalar('train/score', stats['score'], episode)
            writer.add_scalar('train/max_tile', stats['max_tile'], episode)
            writer.add_scalar('train/steps', stats['steps'], episode)
            writer.add_scalar('train/epsilon', stats['epsilon'], episode)
            writer.add_scalar('train/avg_loss', stats['avg_loss'], episode)
            writer.add_scalar('train/avg_score_100', recent_avg, episode)
        
        # Evaluation
        if (episode + 1) % args.eval_freq == 0:
            eval_stats = evaluate(Game, agent, num_episodes=args.eval_episodes, seed=args.seed + 1000000)
            
            print(f"\n[Eval @ {episode + 1}] Avg Score: {eval_stats['avg_score']:.0f} ± {eval_stats['std_score']:.0f}, "
                  f"Max Score: {eval_stats['max_score']}, Avg Max Tile: {eval_stats['avg_max_tile']:.0f}")
            print(f"Tile distribution: {eval_stats['tile_distribution']}")
            
            if writer:
                writer.add_scalar('eval/avg_score', eval_stats['avg_score'], episode)
                writer.add_scalar('eval/max_score', eval_stats['max_score'], episode)
                writer.add_scalar('eval/avg_max_tile', eval_stats['avg_max_tile'], episode)
            
            # Save best model
            if eval_stats['avg_score'] > best_avg_score:
                best_avg_score = eval_stats['avg_score']
                agent.save(os.path.join(args.checkpoint_dir, 'best_model.pt'))
                print(f"New best model saved! Avg score: {best_avg_score:.0f}")
        
        # Periodic checkpoint
        if (episode + 1) % args.save_freq == 0:
            agent.save(os.path.join(args.checkpoint_dir, f'checkpoint_{episode + 1}.pt'))
    
    # Final save
    agent.save(os.path.join(args.checkpoint_dir, 'final_model.pt'))
    
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed / 3600:.1f} hours")
    print(f"Best average score: {best_avg_score:.0f}")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
