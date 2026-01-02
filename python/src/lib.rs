//! # 2048 Python Bindings
//!
//! This crate provides Python bindings to the 2048 game engine using PyO3.
//! It exposes a `Game` class that can be used for reinforcement learning training.

use game_2048_core::{Action, Game as CoreGame, StepResult};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

/// Python wrapper for the 2048 game.
///
/// Usage:
///     from game_2048 import Game
///     game = Game(seed=42)
///     result = game.step(2)  # Left
///     print(result)  # {'board': [...], 'score': 0, 'reward': 0, 'changed': True, 'done': False}
#[pyclass]
pub struct Game {
    inner: CoreGame,
}

#[pymethods]
impl Game {
    /// Create a new game with the given seed.
    ///
    /// Args:
    ///     seed: 64-bit seed for deterministic RNG
    #[new]
    fn new(seed: u64) -> Self {
        Game {
            inner: CoreGame::new(seed),
        }
    }

    /// Reset the game to initial state with a new seed.
    ///
    /// Args:
    ///     seed: 64-bit seed for deterministic RNG
    fn reset(&mut self, seed: u64) {
        self.inner.reset(seed);
    }

    /// Execute a move in the given direction.
    ///
    /// Args:
    ///     action: 0=Up, 1=Down, 2=Left, 3=Right
    ///
    /// Returns:
    ///     dict with keys: board, score, reward, changed, done
    fn step(&mut self, py: Python<'_>, action: u8) -> PyResult<PyObject> {
        let action = match Action::from_u8(action) {
            Some(a) => a,
            None => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid action: {}. Must be 0-3 (Up, Down, Left, Right).",
                    action
                )));
            }
        };

        let result = self.inner.step(action);
        self.create_result_dict(py, result)
    }

    /// Get the current board state.
    ///
    /// Returns:
    ///     List of 16 integers (row-major order)
    fn board(&self) -> Vec<u16> {
        self.inner.board().to_vec()
    }

    /// Get the current score.
    fn score(&self) -> u32 {
        self.inner.score()
    }

    /// Check if the game is over.
    fn is_done(&self) -> bool {
        self.inner.is_done()
    }

    /// Get the maximum tile value on the board.
    fn max_tile(&self) -> u16 {
        self.inner.max_tile()
    }

    /// Get legal actions as a list of 4 booleans [Up, Down, Left, Right].
    fn legal_actions(&self) -> Vec<bool> {
        self.inner.legal_actions().to_vec()
    }

    /// Get the number of empty cells on the board.
    fn empty_count(&self) -> usize {
        self.inner.empty_count()
    }

    /// Get a normalized board state for neural network input.
    ///
    /// Returns:
    ///     List of 16 floats where each tile value is log2(value)/17 or 0
    fn normalized_board(&self) -> Vec<f32> {
        self.inner
            .board()
            .iter()
            .map(|&v| {
                if v == 0 {
                    0.0
                } else {
                    (v as f32).log2() / 17.0
                }
            })
            .collect()
    }

    /// Get one-hot encoded board state for neural network input.
    ///
    /// Returns:
    ///     List of 16 * 18 = 288 floats (18 channels for values 0, 2, 4, ..., 2^17)
    fn one_hot_board(&self) -> Vec<f32> {
        let mut result = vec![0.0f32; 16 * 18];
        for (i, &v) in self.inner.board().iter().enumerate() {
            let channel = if v == 0 {
                0
            } else {
                ((v as f32).log2() as usize).min(17)
            };
            result[i * 18 + channel] = 1.0;
        }
        result
    }

    fn __repr__(&self) -> String {
        format!(
            "Game(score={}, max_tile={}, done={})",
            self.inner.score(),
            self.inner.max_tile(),
            self.inner.is_done()
        )
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }
}

impl Game {
    fn create_result_dict(&self, py: Python<'_>, result: StepResult) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("board", self.inner.board().to_vec())?;
        dict.set_item("score", self.inner.score())?;
        dict.set_item("reward", result.reward)?;
        dict.set_item("changed", result.changed)?;
        dict.set_item("done", result.done)?;
        Ok(dict.into())
    }
}

/// Action constants for convenience.
#[pyclass]
struct Actions;

#[pymethods]
impl Actions {
    #[classattr]
    const UP: u8 = 0;
    #[classattr]
    const DOWN: u8 = 1;
    #[classattr]
    const LEFT: u8 = 2;
    #[classattr]
    const RIGHT: u8 = 3;
}

/// Python module for the 2048 game.
#[pymodule]
fn game_2048(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Game>()?;
    m.add_class::<Actions>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_creation() {
        let game = Game::new(42);
        assert!(!game.is_done());
    }

    #[test]
    fn test_determinism() {
        let game1 = Game::new(12345);
        let game2 = Game::new(12345);
        assert_eq!(game1.board(), game2.board());
    }
}
