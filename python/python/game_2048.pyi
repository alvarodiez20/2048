"""Python type hints for game_2048 Rust extension."""

from typing import List, Dict


class Actions:
    """Action constants for move directions."""
    UP: int
    DOWN: int
    LEFT: int
    RIGHT: int


class Game:
    """2048 game engine with deterministic, seedable RNG.
    
    Args:
        seed: 64-bit seed for reproducible gameplay
    
    Example:
        >>> game = Game(seed=42)
        >>> result = game.step(2)  # Left
        >>> print(game.score())
    """
    
    def __init__(self, seed: int) -> None: ...
    
    def reset(self, seed: int) -> None:
        """Reset game to initial state with new seed."""
        ...
    
    def step(self, action: int) -> Dict[str, object]:
        """Execute a move (0=Up, 1=Down, 2=Left, 3=Right).
        
        Returns:
            dict with keys: board, score, reward, changed, done
        """
        ...
    
    def board(self) -> List[int]:
        """Get current board state (16 values, row-major order)."""
        ...
    
    def score(self) -> int:
        """Get current total score."""
        ...
    
    def is_done(self) -> bool:
        """Check if game is over (no valid moves)."""
        ...
    
    def max_tile(self) -> int:
        """Get the highest tile value on the board."""
        ...
    
    def legal_actions(self) -> List[bool]:
        """Get valid moves as [Up, Down, Left, Right] booleans."""
        ...
    
    def empty_count(self) -> int:
        """Get number of empty cells on the board."""
        ...
    
    def normalized_board(self) -> List[float]:
        """Get board as log2-normalized floats for neural network input.
        
        Each tile value is converted to log2(value)/17, or 0 for empty.
        """
        ...
    
    def one_hot_board(self) -> List[float]:
        """Get board as one-hot encoding (16 * 18 = 288 values).
        
        18 channels for values: 0, 2, 4, 8, ..., 2^17
        """
        ...
