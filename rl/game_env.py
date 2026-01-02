"""
2048 Game Environment - Pure Python Implementation

This module provides a pure Python 2048 game environment for RL training.
It mirrors the Rust core engine logic exactly for consistency.

For better performance, install the Rust Python bindings:
    cd ../python && maturin develop
Then import: from game_2048 import Game
"""

import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class StepResult:
    """Result of a game step."""
    board: List[int]
    score: int
    reward: int
    changed: bool
    done: bool


class Game:
    """
    2048 game environment with deterministic, seedable RNG.
    
    This is a pure Python implementation that matches the Rust core engine.
    Use this for training when the Rust bindings aren't available.
    """
    
    # Action constants
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    ACTION_NAMES = ['Up', 'Down', 'Left', 'Right']
    
    def __init__(self, seed: int = 42):
        """Create a new game with the given seed."""
        self._seed = seed
        self._rng = random.Random(seed)
        self.board: List[int] = [0] * 16
        self.score: int = 0
        self._done: bool = False
        
        # Spawn initial tiles
        self._spawn_tile()
        self._spawn_tile()
        self._update_done()
    
    def reset(self, seed: Optional[int] = None) -> List[int]:
        """Reset the game to initial state."""
        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)
        self.board = [0] * 16
        self.score = 0
        self._done = False
        
        self._spawn_tile()
        self._spawn_tile()
        self._update_done()
        
        return self.board.copy()
    
    def step(self, action: int) -> Dict:
        """
        Execute a move in the given direction.
        
        Args:
            action: 0=Up, 1=Down, 2=Left, 3=Right
        
        Returns:
            Dict with keys: board, score, reward, changed, done
        """
        if self._done:
            return {
                'board': self.board.copy(),
                'score': self.score,
                'reward': 0,
                'changed': False,
                'done': True
            }
        
        old_board = self.board.copy()
        reward = self._apply_move(action)
        
        changed = self.board != old_board
        if changed:
            self.score += reward
            self._spawn_tile()
        
        self._update_done()
        
        return {
            'board': self.board.copy(),
            'score': self.score,
            'reward': reward,
            'changed': changed,
            'done': self._done
        }
    
    def is_done(self) -> bool:
        """Check if the game is over."""
        return self._done
    
    def legal_actions(self) -> List[bool]:
        """Get legal actions as [Up, Down, Left, Right] booleans."""
        return [self._can_move(i) for i in range(4)]
    
    def max_tile(self) -> int:
        """Get the maximum tile value."""
        return max(self.board)
    
    def empty_count(self) -> int:
        """Get the number of empty cells."""
        return self.board.count(0)
    
    def normalized_board(self) -> List[float]:
        """Get board as log2-normalized floats for neural network input."""
        import math
        return [
            0.0 if v == 0 else math.log2(v) / 17.0
            for v in self.board
        ]
    
    def get_state(self) -> List[float]:
        """Get the normalized state for neural network input."""
        return self.normalized_board()
    
    def _spawn_tile(self) -> None:
        """Spawn a new tile (90% 2, 10% 4) in a random empty cell."""
        empty_cells = [i for i, v in enumerate(self.board) if v == 0]
        if not empty_cells:
            return
        
        idx = self._rng.choice(empty_cells)
        value = 2 if self._rng.random() < 0.9 else 4
        self.board[idx] = value
    
    def _update_done(self) -> None:
        """Update the done flag."""
        self._done = not any(self._can_move(i) for i in range(4))
    
    def _can_move(self, action: int) -> bool:
        """Check if a move would change the board."""
        test_board = self.board.copy()
        self._apply_move_to_board(test_board, action)
        return test_board != self.board
    
    def _apply_move(self, action: int) -> int:
        """Apply a move to the board and return the reward."""
        return self._apply_move_to_board(self.board, action)
    
    def _apply_move_to_board(self, board: List[int], action: int) -> int:
        """Apply a move to a board array and return the reward."""
        total_reward = 0
        
        if action == self.LEFT:
            for row in range(4):
                start = row * 4
                line = board[start:start + 4]
                reward = self._compress_and_merge(line)
                board[start:start + 4] = line
                total_reward += reward
        
        elif action == self.RIGHT:
            for row in range(4):
                start = row * 4
                line = board[start:start + 4][::-1]
                reward = self._compress_and_merge(line)
                board[start:start + 4] = line[::-1]
                total_reward += reward
        
        elif action == self.UP:
            for col in range(4):
                line = [board[col + i * 4] for i in range(4)]
                reward = self._compress_and_merge(line)
                for i in range(4):
                    board[col + i * 4] = line[i]
                total_reward += reward
        
        elif action == self.DOWN:
            for col in range(4):
                line = [board[col + i * 4] for i in range(3, -1, -1)]
                reward = self._compress_and_merge(line)
                for i in range(4):
                    board[col + (3 - i) * 4] = line[i]
                total_reward += reward
        
        return total_reward
    
    @staticmethod
    def _compress_and_merge(line: List[int]) -> int:
        """Compress and merge a line of 4 tiles. Returns reward."""
        # Step 1: Compress
        Game._compress(line)
        
        # Step 2: Merge
        reward = 0
        for i in range(3):
            if line[i] != 0 and line[i] == line[i + 1]:
                line[i] *= 2
                reward += line[i]
                line[i + 1] = 0
        
        # Step 3: Compress again
        Game._compress(line)
        
        return reward
    
    @staticmethod
    def _compress(line: List[int]) -> None:
        """Compress a line by moving non-zero values to the front."""
        write_idx = 0
        for read_idx in range(4):
            if line[read_idx] != 0:
                if write_idx != read_idx:
                    line[write_idx] = line[read_idx]
                    line[read_idx] = 0
                write_idx += 1
    
    def __repr__(self) -> str:
        return f"Game(score={self.score}, max_tile={self.max_tile()}, done={self._done})"
    
    def __str__(self) -> str:
        lines = [f"Score: {self.score}"]
        lines.append("+------+------+------+------+")
        for row in range(4):
            cells = []
            for col in range(4):
                val = self.board[row * 4 + col]
                if val == 0:
                    cells.append("      ")
                else:
                    cells.append(f"{val:^6}")
            lines.append("|" + "|".join(cells) + "|")
            lines.append("+------+------+------+------+")
        return "\n".join(lines)


# Try to import the faster Rust implementation
try:
    from game_2048 import Game as RustGame, Actions
    
    # Wrap Rust Game to have same interface
    class FastGame:
        """Fast Rust-backed Game wrapper."""
        UP = Actions.UP
        DOWN = Actions.DOWN
        LEFT = Actions.LEFT
        RIGHT = Actions.RIGHT
        
        def __init__(self, seed: int = 42):
            self._game = RustGame(seed)
            self._seed = seed
        
        def reset(self, seed: int = None) -> List[int]:
            if seed is not None:
                self._seed = seed
            self._game.reset(self._seed)
            return list(self._game.board())
        
        def step(self, action: int) -> Dict:
            return self._game.step(action)
        
        @property
        def board(self) -> List[int]:
            return list(self._game.board())
        
        @property
        def score(self) -> int:
            return self._game.score()
        
        def is_done(self) -> bool:
            return self._game.is_done()
        
        def legal_actions(self) -> List[bool]:
            return list(self._game.legal_actions())
        
        def max_tile(self) -> int:
            return self._game.max_tile()
        
        def normalized_board(self) -> List[float]:
            return list(self._game.normalized_board())
        
        def get_state(self) -> List[float]:
            return self.normalized_board()
    
    # Use Fast version if available
    GameEnv = FastGame
    print("Using fast Rust-backed Game implementation")
    
except ImportError:
    # Use pure Python version
    GameEnv = Game
    print("Using pure Python Game implementation (install Rust bindings for better performance)")


if __name__ == "__main__":
    # Quick test
    game = Game(seed=42)
    print(game)
    
    # Play a few moves
    for action in [Game.LEFT, Game.UP, Game.RIGHT, Game.DOWN]:
        result = game.step(action)
        print(f"\nAction: {Game.ACTION_NAMES[action]}")
        print(f"Reward: {result['reward']}, Changed: {result['changed']}")
        print(game)
        
        if game.is_done():
            print("Game Over!")
            break
