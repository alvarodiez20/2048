# game-2048 - Python Bindings for 2048 Game Engine

Python bindings for the high-performance 2048 game engine written in Rust. Perfect for reinforcement learning research and training AI agents.

## Installation

### From Source (requires Rust)

```bash
# Install maturin
pip install maturin

# Build and install
cd python
maturin develop

# Or build a wheel
maturin build --release
pip install target/wheels/game_2048-*.whl
```

## Quick Start

```python
from game_2048 import Game, Actions

# Create a new game with seed for reproducibility
game = Game(seed=42)

# Print initial state
print(game)

# Make a move (0=Up, 1=Down, 2=Left, 3=Right)
result = game.step(Actions.LEFT)
print(f"Score: {result['score']}, Reward: {result['reward']}")

# Check game state
print(f"Board: {game.board()}")
print(f"Legal actions: {game.legal_actions()}")
print(f"Game over: {game.is_done()}")
```

## API Reference

### `Game(seed: int)`

Create a new 2048 game with deterministic seed.

#### Methods

- `step(action: int) -> dict`: Execute a move. Returns `{board, score, reward, changed, done}`
- `reset(seed: int)`: Reset game to initial state with new seed
- `board() -> list[int]`: Get current board (16 values, row-major order)
- `score() -> int`: Get current score
- `is_done() -> bool`: Check if game is over
- `max_tile() -> int`: Get highest tile value
- `legal_actions() -> list[bool]`: Get valid moves `[Up, Down, Left, Right]`
- `empty_count() -> int`: Get number of empty cells
- `normalized_board() -> list[float]`: Board as log2-normalized floats (for NN input)
- `one_hot_board() -> list[float]`: Board as one-hot encoding (288 values)

### `Actions`

Constants for move directions:
- `Actions.UP = 0`
- `Actions.DOWN = 1`
- `Actions.LEFT = 2`
- `Actions.RIGHT = 3`

## Usage for Reinforcement Learning

```python
import random
from game_2048 import Game, Actions

def train_episode(seed: int):
    game = Game(seed=seed)
    total_reward = 0
    steps = 0
    
    while not game.is_done():
        # Get valid actions
        legal = game.legal_actions()
        valid_actions = [i for i, v in enumerate(legal) if v]
        
        if not valid_actions:
            break
        
        # Your policy here (random for example)
        action = random.choice(valid_actions)
        
        # Get normalized state for neural network
        state = game.normalized_board()
        
        # Take action
        result = game.step(action)
        total_reward += result['reward']
        steps += 1
    
    return game.score(), game.max_tile(), steps

# Run training episodes
for episode in range(1000):
    score, max_tile, steps = train_episode(seed=episode)
    print(f"Episode {episode}: Score={score}, MaxTile={max_tile}, Steps={steps}")
```

## Features

- **Deterministic**: Same seed always produces same game
- **Fast**: Rust core engine, minimal overhead
- **RL-ready**: Normalized board states, reward signals
- **Typed**: Full type hints available
