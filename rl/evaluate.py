"""
Evaluation script for trained 2048 DQN agents.

Usage:
    python evaluate.py --model checkpoints/best_model.pt --episodes 100
"""

import argparse
import numpy as np
from typing import Dict, List

from game_env import Game
from dqn_agent import DQNAgent


def evaluate(agent: DQNAgent, num_episodes: int = 100, seed: int = 0, 
             verbose: bool = False) -> Dict:
    """
    Evaluate the agent.
    
    Returns:
        Dict with evaluation statistics
    """
    scores = []
    max_tiles = []
    steps_list = []
    
    for i in range(num_episodes):
        game = Game(seed=seed + i)
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
        
        if verbose:
            print(f"Episode {i + 1}: Score={game.score}, MaxTile={game.max_tile()}, Steps={steps}")
    
    # Tile distribution
    tile_counts = {}
    for tile in max_tiles:
        tile_counts[tile] = tile_counts.get(tile, 0) + 1
    
    # Calculate percentiles
    percentiles = np.percentile(scores, [25, 50, 75, 90, 95])
    
    return {
        'num_episodes': num_episodes,
        'avg_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': np.max(scores),
        'min_score': np.min(scores),
        'median_score': np.median(scores),
        'p25_score': percentiles[0],
        'p75_score': percentiles[2],
        'p90_score': percentiles[3],
        'p95_score': percentiles[4],
        'avg_max_tile': np.mean(max_tiles),
        'avg_steps': np.mean(steps_list),
        'tile_distribution': tile_counts,
        'scores': scores,
        'max_tiles': max_tiles
    }


def evaluate_random(num_episodes: int = 100, seed: int = 0) -> Dict:
    """Evaluate random policy baseline."""
    import random
    random.seed(seed)
    
    scores = []
    max_tiles = []
    
    for i in range(num_episodes):
        game = Game(seed=seed + i)
        steps = 0
        
        while not game.is_done() and steps < 10000:
            legal_actions = game.legal_actions()
            valid_actions = [j for j, legal in enumerate(legal_actions) if legal]
            if not valid_actions:
                break
            action = random.choice(valid_actions)
            game.step(action)
            steps += 1
        
        scores.append(game.score)
        max_tiles.append(game.max_tile())
    
    tile_counts = {}
    for tile in max_tiles:
        tile_counts[tile] = tile_counts.get(tile, 0) + 1
    
    return {
        'avg_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': np.max(scores),
        'avg_max_tile': np.mean(max_tiles),
        'tile_distribution': tile_counts
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate 2048 DQN agent')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Print each episode')
    parser.add_argument('--compare-random', action='store_true', help='Compare with random baseline')
    args = parser.parse_args()
    
    # Load agent
    print(f"Loading model from {args.model}...")
    agent = DQNAgent()
    agent.load(args.model)
    print(f"Model loaded (trained for {agent.steps} steps)")
    
    # Evaluate
    print(f"\nEvaluating {args.episodes} episodes...")
    stats = evaluate(agent, num_episodes=args.episodes, seed=args.seed, verbose=args.verbose)
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Episodes:      {stats['num_episodes']}")
    print(f"Average Score: {stats['avg_score']:.1f} ± {stats['std_score']:.1f}")
    print(f"Median Score:  {stats['median_score']:.1f}")
    print(f"Min/Max Score: {stats['min_score']} / {stats['max_score']}")
    print(f"Percentiles:   25th={stats['p25_score']:.0f}, 75th={stats['p75_score']:.0f}, 90th={stats['p90_score']:.0f}")
    print(f"Avg Max Tile:  {stats['avg_max_tile']:.1f}")
    print(f"Avg Steps:     {stats['avg_steps']:.1f}")
    print("\nTile Distribution:")
    for tile in sorted(stats['tile_distribution'].keys()):
        count = stats['tile_distribution'][tile]
        pct = count / stats['num_episodes'] * 100
        print(f"  {tile:5d}: {count:4d} ({pct:5.1f}%)")
    
    # Compare with random
    if args.compare_random:
        print("\n" + "-" * 50)
        print("RANDOM BASELINE")
        print("-" * 50)
        random_stats = evaluate_random(num_episodes=args.episodes, seed=args.seed)
        print(f"Average Score: {random_stats['avg_score']:.1f} ± {random_stats['std_score']:.1f}")
        print(f"Max Score:     {random_stats['max_score']}")
        print(f"Avg Max Tile:  {random_stats['avg_max_tile']:.1f}")
        print("\nTile Distribution:")
        for tile in sorted(random_stats['tile_distribution'].keys()):
            count = random_stats['tile_distribution'][tile]
            pct = count / args.episodes * 100
            print(f"  {tile:5d}: {count:4d} ({pct:5.1f}%)")
        
        improvement = (stats['avg_score'] - random_stats['avg_score']) / random_stats['avg_score'] * 100
        print(f"\nImprovement over random: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
