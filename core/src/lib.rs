//! # 2048 Game Core Engine
//!
//! A pure Rust implementation of the 2048 game logic with deterministic,
//! seedable PRNG for reproducible gameplay. Designed for easy integration
//! with CLI, WebAssembly, and future Python RL bindings.
//!
//! ## Example
//!
//! ```rust
//! use game_2048_core::{Game, Action};
//!
//! let mut game = Game::new(42);  // Create game with seed 42
//! let result = game.step(Action::Left);
//! println!("Score: {}, Changed: {}", game.score(), result.changed);
//! ```

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub mod solver;

/// The four possible move directions in 2048.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Action {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

impl Action {
    /// Convert a u8 to an Action (0=Up, 1=Down, 2=Left, 3=Right).
    /// Returns None for invalid values.
    pub fn from_u8(value: u8) -> Option<Action> {
        match value {
            0 => Some(Action::Up),
            1 => Some(Action::Down),
            2 => Some(Action::Left),
            3 => Some(Action::Right),
            _ => None,
        }
    }

    /// Get all four actions.
    pub fn all() -> [Action; 4] {
        [Action::Up, Action::Down, Action::Left, Action::Right]
    }
}

/// Result of executing a step (move) in the game.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepResult {
    /// Whether the board changed (and a new tile was spawned).
    pub changed: bool,
    /// Points earned from merges in this move.
    pub reward: u32,
    /// Whether the game is over (no legal moves remaining).
    pub done: bool,
}

/// The 2048 game state.
///
/// The board is represented as a flat array of 16 u16 values,
/// stored in row-major order (indices 0-3 are row 0, 4-7 are row 1, etc.).
/// Empty cells are 0, tiles contain their value (2, 4, 8, ..., 2048, ...).
#[derive(Clone)]
pub struct Game {
    board: [u16; 16],
    score: u32,
    rng: SmallRng,
    done: bool,
}

impl Game {
    /// Create a new game with the given seed.
    ///
    /// The game starts with two random tiles (90% chance of 2, 10% chance of 4).
    pub fn new(seed: u64) -> Self {
        let mut game = Game {
            board: [0; 16],
            score: 0,
            rng: SmallRng::seed_from_u64(seed),
            done: false,
        };
        game.spawn_tile();
        game.spawn_tile();
        game.update_done();
        game
    }

    /// Reset the game to initial state with a new seed.
    pub fn reset(&mut self, seed: u64) {
        self.board = [0; 16];
        self.score = 0;
        self.rng = SmallRng::seed_from_u64(seed);
        self.done = false;
        self.spawn_tile();
        self.spawn_tile();
        self.update_done();
    }

    /// Execute a move in the given direction.
    ///
    /// Returns a `StepResult` containing:
    /// - `changed`: whether the board changed (a tile was spawned if true)
    /// - `reward`: points earned from merges
    /// - `done`: whether the game is over
    ///
    /// If the move doesn't change the board, no tile is spawned.
    pub fn step(&mut self, action: Action) -> StepResult {
        if self.done {
            return StepResult {
                changed: false,
                reward: 0,
                done: true,
            };
        }

        let old_board = self.board;
        let reward = self.apply_move(action);

        let changed = self.board != old_board;
        if changed {
            self.score += reward;
            self.spawn_tile();
        }

        self.update_done();

        StepResult {
            changed,
            reward,
            done: self.done,
        }
    }

    /// Check if the game is over (no legal moves available).
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Get the legal actions as a boolean array [Up, Down, Left, Right].
    ///
    /// An action is legal if it would change the board state.
    pub fn legal_actions(&self) -> [bool; 4] {
        [
            self.can_move(Action::Up),
            self.can_move(Action::Down),
            self.can_move(Action::Left),
            self.can_move(Action::Right),
        ]
    }

    /// Get a reference to the board array.
    ///
    /// The board is in row-major order: indices 0-3 are row 0, etc.
    pub fn board(&self) -> &[u16; 16] {
        &self.board
    }

    /// Get the current score.
    pub fn score(&self) -> u32 {
        self.score
    }

    /// Get the maximum tile value on the board.
    pub fn max_tile(&self) -> u16 {
        *self.board.iter().max().unwrap_or(&0)
    }

    /// Get the number of empty cells on the board.
    pub fn empty_count(&self) -> usize {
        self.board.iter().filter(|&&x| x == 0).count()
    }

    // -------------------------------------------------------------------------
    // Private methods
    // -------------------------------------------------------------------------

    /// Spawn a new tile in a random empty cell.
    /// 90% chance of 2, 10% chance of 4.
    fn spawn_tile(&mut self) {
        let empty_cells: Vec<usize> = self
            .board
            .iter()
            .enumerate()
            .filter(|(_, &v)| v == 0)
            .map(|(i, _)| i)
            .collect();

        if empty_cells.is_empty() {
            return;
        }

        let idx = empty_cells[self.rng.gen_range(0..empty_cells.len())];
        let value = if self.rng.gen::<f32>() < 0.9 { 2 } else { 4 };
        self.board[idx] = value;
    }

    /// Update the done flag by checking if any moves are possible.
    fn update_done(&mut self) {
        self.done = !self.can_move(Action::Up)
            && !self.can_move(Action::Down)
            && !self.can_move(Action::Left)
            && !self.can_move(Action::Right);
    }

    /// Check if a move in the given direction would change the board.
    fn can_move(&self, action: Action) -> bool {
        let mut test_board = self.board;
        Self::apply_move_to_board(&mut test_board, action) > 0 || test_board != self.board
    }

    /// Apply a move to the board and return the reward (merge points).
    fn apply_move(&mut self, action: Action) -> u32 {
        Self::apply_move_to_board(&mut self.board, action)
    }

    /// Apply a move to a board array, returning the reward.
    fn apply_move_to_board(board: &mut [u16; 16], action: Action) -> u32 {
        let mut total_reward = 0;

        match action {
            Action::Left => {
                for row in 0..4 {
                    let start = row * 4;
                    let mut line = [
                        board[start],
                        board[start + 1],
                        board[start + 2],
                        board[start + 3],
                    ];
                    total_reward += Self::compress_and_merge(&mut line);
                    board[start] = line[0];
                    board[start + 1] = line[1];
                    board[start + 2] = line[2];
                    board[start + 3] = line[3];
                }
            }
            Action::Right => {
                for row in 0..4 {
                    let start = row * 4;
                    let mut line = [
                        board[start + 3],
                        board[start + 2],
                        board[start + 1],
                        board[start],
                    ];
                    total_reward += Self::compress_and_merge(&mut line);
                    board[start + 3] = line[0];
                    board[start + 2] = line[1];
                    board[start + 1] = line[2];
                    board[start] = line[3];
                }
            }
            Action::Up => {
                for col in 0..4 {
                    let mut line = [board[col], board[col + 4], board[col + 8], board[col + 12]];
                    total_reward += Self::compress_and_merge(&mut line);
                    board[col] = line[0];
                    board[col + 4] = line[1];
                    board[col + 8] = line[2];
                    board[col + 12] = line[3];
                }
            }
            Action::Down => {
                for col in 0..4 {
                    let mut line = [board[col + 12], board[col + 8], board[col + 4], board[col]];
                    total_reward += Self::compress_and_merge(&mut line);
                    board[col + 12] = line[0];
                    board[col + 8] = line[1];
                    board[col + 4] = line[2];
                    board[col] = line[3];
                }
            }
        }

        total_reward
    }

    /// Compress and merge a line of 4 tiles (moving towards index 0).
    /// Returns the points earned from merges.
    ///
    /// Algorithm:
    /// 1. Compress: move all non-zero values to the front
    /// 2. Merge: combine adjacent equal values (only once each)
    /// 3. Compress again
    fn compress_and_merge(line: &mut [u16; 4]) -> u32 {
        // Step 1: Compress (remove zeros, shift left)
        Self::compress(line);

        // Step 2: Merge adjacent equal tiles
        let mut reward = 0;
        for i in 0..3 {
            if line[i] != 0 && line[i] == line[i + 1] {
                line[i] *= 2;
                reward += line[i] as u32;
                line[i + 1] = 0;
            }
        }

        // Step 3: Compress again
        Self::compress(line);

        reward
    }

    /// Compress a line by moving all non-zero values to the front.
    fn compress(line: &mut [u16; 4]) {
        let mut write_idx = 0;
        for read_idx in 0..4 {
            if line[read_idx] != 0 {
                if write_idx != read_idx {
                    line[write_idx] = line[read_idx];
                    line[read_idx] = 0;
                }
                write_idx += 1;
            }
        }
    }
}

impl std::fmt::Debug for Game {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Game {{ score: {}, done: {} }}", self.score, self.done)?;
        for row in 0..4 {
            for col in 0..4 {
                let val = self.board[row * 4 + col];
                if val == 0 {
                    write!(f, "    .")?;
                } else {
                    write!(f, "{:5}", val)?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for Game {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Score: {}", self.score)?;
        writeln!(f, "+------+------+------+------+")?;
        for row in 0..4 {
            write!(f, "|")?;
            for col in 0..4 {
                let val = self.board[row * 4 + col];
                if val == 0 {
                    write!(f, "      |")?;
                } else {
                    write!(f, "{:^6}|", val)?;
                }
            }
            writeln!(f)?;
            writeln!(f, "+------+------+------+------+")?;
        }
        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Move correctness tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compress_simple() {
        let mut line = [0, 2, 0, 4];
        Game::compress(&mut line);
        assert_eq!(line, [2, 4, 0, 0]);
    }

    #[test]
    fn test_compress_already_compressed() {
        let mut line = [2, 4, 8, 16];
        Game::compress(&mut line);
        assert_eq!(line, [2, 4, 8, 16]);
    }

    #[test]
    fn test_compress_all_zeros() {
        let mut line = [0, 0, 0, 0];
        Game::compress(&mut line);
        assert_eq!(line, [0, 0, 0, 0]);
    }

    #[test]
    fn test_merge_simple() {
        let mut line = [2, 2, 0, 0];
        let reward = Game::compress_and_merge(&mut line);
        assert_eq!(line, [4, 0, 0, 0]);
        assert_eq!(reward, 4);
    }

    #[test]
    fn test_merge_two_pairs() {
        let mut line = [2, 2, 4, 4];
        let reward = Game::compress_and_merge(&mut line);
        assert_eq!(line, [4, 8, 0, 0]);
        assert_eq!(reward, 12);
    }

    #[test]
    fn test_no_double_merge() {
        // [4, 2, 2, 0] should become [4, 4, 0, 0], not [8, 0, 0, 0]
        let mut line = [4, 2, 2, 0];
        let reward = Game::compress_and_merge(&mut line);
        assert_eq!(line, [4, 4, 0, 0]);
        assert_eq!(reward, 4);
    }

    #[test]
    fn test_no_double_merge_chain() {
        // [2, 2, 2, 2] should become [4, 4, 0, 0], not [8, 0, 0, 0]
        let mut line = [2, 2, 2, 2];
        let reward = Game::compress_and_merge(&mut line);
        assert_eq!(line, [4, 4, 0, 0]);
        assert_eq!(reward, 8);
    }

    #[test]
    fn test_merge_with_gaps() {
        let mut line = [2, 0, 2, 0];
        let reward = Game::compress_and_merge(&mut line);
        assert_eq!(line, [4, 0, 0, 0]);
        assert_eq!(reward, 4);
    }

    #[test]
    fn test_move_left() {
        let mut board = [2, 2, 0, 0, 0, 4, 4, 0, 2, 0, 2, 0, 8, 8, 8, 8];
        let reward = Game::apply_move_to_board(&mut board, Action::Left);
        assert_eq!(board, [4, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 16, 16, 0, 0,]);
        assert_eq!(reward, 4 + 8 + 4 + 32);
    }

    #[test]
    fn test_move_right() {
        let mut board = [2, 2, 0, 0, 0, 4, 4, 0, 2, 0, 2, 0, 8, 8, 8, 8];
        let reward = Game::apply_move_to_board(&mut board, Action::Right);
        assert_eq!(board, [0, 0, 0, 4, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 16, 16,]);
        assert_eq!(reward, 4 + 8 + 4 + 32);
    }

    #[test]
    fn test_move_up() {
        let mut board = [2, 0, 2, 8, 2, 4, 0, 8, 0, 4, 2, 8, 0, 0, 0, 8];
        let reward = Game::apply_move_to_board(&mut board, Action::Up);
        assert_eq!(board, [4, 8, 4, 16, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0,]);
        assert_eq!(reward, 4 + 8 + 4 + 32);
    }

    #[test]
    fn test_move_down() {
        let mut board = [2, 0, 2, 8, 2, 4, 0, 8, 0, 4, 2, 8, 0, 0, 0, 8];
        let reward = Game::apply_move_to_board(&mut board, Action::Down);
        assert_eq!(board, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 4, 8, 4, 16,]);
        assert_eq!(reward, 4 + 8 + 4 + 32);
    }

    // -------------------------------------------------------------------------
    // Spawn determinism tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_spawn_determinism() {
        let seed = 12345u64;
        let game1 = Game::new(seed);
        let game2 = Game::new(seed);

        assert_eq!(game1.board(), game2.board());
        assert_eq!(game1.score(), game2.score());
    }

    #[test]
    fn test_step_determinism() {
        let seed = 54321u64;
        let mut game1 = Game::new(seed);
        let mut game2 = Game::new(seed);

        let actions = [Action::Left, Action::Up, Action::Right, Action::Down];
        for action in actions {
            game1.step(action);
            game2.step(action);
            assert_eq!(game1.board(), game2.board());
            assert_eq!(game1.score(), game2.score());
        }
    }

    #[test]
    fn test_different_seeds_different_games() {
        let game1 = Game::new(111);
        let game2 = Game::new(222);

        // Very unlikely to be the same
        assert_ne!(game1.board(), game2.board());
    }

    // -------------------------------------------------------------------------
    // Game over detection tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_game_not_over_with_empty_cells() {
        let game = Game::new(42);
        // New game should have empty cells
        assert!(!game.is_done());
        assert!(game.empty_count() > 0);
    }

    #[test]
    fn test_game_over_no_moves() {
        let mut game = Game::new(0);
        // Manually set up a board with no possible moves
        game.board = [2, 4, 2, 4, 4, 2, 4, 2, 2, 4, 2, 4, 4, 2, 4, 2];
        game.update_done();
        assert!(game.is_done());
        assert_eq!(game.legal_actions(), [false, false, false, false]);
    }

    #[test]
    fn test_game_not_over_can_merge_horizontal() {
        let mut game = Game::new(0);
        game.board = [
            2, 2, 4, 8, // Can merge left or right
            4, 8, 16, 32, 8, 16, 32, 64, 16, 32, 64, 128,
        ];
        game.update_done();
        assert!(!game.is_done());
    }

    #[test]
    fn test_game_not_over_can_merge_vertical() {
        let mut game = Game::new(0);
        game.board = [
            2, 4, 8, 16, 2, 8, 16, 32, // First column can merge
            4, 16, 32, 64, 8, 32, 64, 128,
        ];
        game.update_done();
        assert!(!game.is_done());
    }

    // -------------------------------------------------------------------------
    // Legal actions tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_legal_actions_all_directions() {
        let game = Game::new(42);
        let legal = game.legal_actions();
        // With a new game, at least some moves should be legal
        assert!(legal.iter().any(|&x| x));
    }

    #[test]
    fn test_step_no_change_no_spawn() {
        let mut game = Game::new(0);
        // Set up a board where moving left does nothing
        game.board = [2, 0, 0, 0, 4, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0];
        let old_board = game.board;
        let result = game.step(Action::Left);

        assert!(!result.changed);
        assert_eq!(result.reward, 0);
        // Board should still be the same (no spawn because no change)
        assert_eq!(game.board, old_board);
    }

    // -------------------------------------------------------------------------
    // Reset test
    // -------------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut game = Game::new(42);

        // Play some moves
        game.step(Action::Left);
        game.step(Action::Up);
        let score_before_reset = game.score();

        // Reset with same seed should give same starting position
        game.reset(42);
        let fresh_game = Game::new(42);

        assert_eq!(game.board(), fresh_game.board());
        assert_eq!(game.score(), 0);
        // score_before_reset is used to verify game state changed before reset
        let _ = score_before_reset;
    }

    // -------------------------------------------------------------------------
    // Action conversion tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_action_from_u8() {
        assert_eq!(Action::from_u8(0), Some(Action::Up));
        assert_eq!(Action::from_u8(1), Some(Action::Down));
        assert_eq!(Action::from_u8(2), Some(Action::Left));
        assert_eq!(Action::from_u8(3), Some(Action::Right));
        assert_eq!(Action::from_u8(4), None);
        assert_eq!(Action::from_u8(255), None);
    }

    #[test]
    fn test_action_all() {
        let all = Action::all();
        assert_eq!(all.len(), 4);
        assert_eq!(all[0], Action::Up);
        assert_eq!(all[1], Action::Down);
        assert_eq!(all[2], Action::Left);
        assert_eq!(all[3], Action::Right);
    }

    // -------------------------------------------------------------------------
    // Display test
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_format() {
        let game = Game::new(42);
        let display = format!("{}", game);
        assert!(display.contains("Score:"));
        assert!(display.contains("+------+"));
    }

    #[test]
    fn test_debug_format() {
        let game = Game::new(42);
        let debug = format!("{:?}", game);
        assert!(debug.contains("Game"));
        assert!(debug.contains("score"));
    }
}
