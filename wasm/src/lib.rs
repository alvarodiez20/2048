//! # 2048 WebAssembly Bindings
//!
//! This crate provides JavaScript-friendly bindings to the 2048 game engine
//! using wasm-bindgen. It wraps the core engine and exposes a class-like API
//! suitable for use in web applications.

use game_2048_core::{Action, Game, StepResult};
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// Result of a step operation, serialized for JavaScript.
#[derive(Serialize)]
pub struct JsStepResult {
    /// The updated board state (16 elements, row-major order).
    pub board: Vec<u16>,
    /// Current total score.
    pub score: u32,
    /// Points earned from this move.
    pub reward: u32,
    /// Whether the board changed.
    pub changed: bool,
    /// Whether the game is over.
    pub done: bool,
}

/// WebAssembly wrapper for the 2048 game.
#[wasm_bindgen]
pub struct WasmGame {
    game: Game,
}

#[wasm_bindgen]
impl WasmGame {
    /// Create a new game with the given seed.
    ///
    /// The seed is a 64-bit integer used to initialize the deterministic RNG.
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u64) -> WasmGame {
        WasmGame {
            game: Game::new(seed),
        }
    }

    /// Reset the game to initial state with a new seed.
    pub fn reset(&mut self, seed: u64) {
        self.game.reset(seed);
    }

    /// Execute a move in the given direction.
    ///
    /// Action values:
    /// - 0 = Up
    /// - 1 = Down
    /// - 2 = Left
    /// - 3 = Right
    ///
    /// Returns a JsValue object containing:
    /// - board: Uint16Array with 16 elements
    /// - score: current total score
    /// - reward: points earned from this move
    /// - changed: whether the board changed
    /// - done: whether the game is over
    pub fn step(&mut self, action: u8) -> JsValue {
        let action = match Action::from_u8(action) {
            Some(a) => a,
            None => {
                // Invalid action, return current state with changed=false
                return self.create_js_result(StepResult {
                    changed: false,
                    reward: 0,
                    done: self.game.is_done(),
                });
            }
        };

        let result = self.game.step(action);
        self.create_js_result(result)
    }

    /// Get the current board state as a JavaScript Uint16Array.
    #[wasm_bindgen(js_name = getBoard)]
    pub fn get_board(&self) -> Vec<u16> {
        self.game.board().to_vec()
    }

    /// Get the current score.
    #[wasm_bindgen(js_name = getScore)]
    pub fn get_score(&self) -> u32 {
        self.game.score()
    }

    /// Check if the game is over.
    #[wasm_bindgen(js_name = isDone)]
    pub fn is_done(&self) -> bool {
        self.game.is_done()
    }

    /// Get the maximum tile value on the board.
    #[wasm_bindgen(js_name = getMaxTile)]
    pub fn get_max_tile(&self) -> u16 {
        self.game.max_tile()
    }

    /// Get legal actions as an array of 4 booleans [Up, Down, Left, Right].
    #[wasm_bindgen(js_name = getLegalActions)]
    pub fn get_legal_actions(&self) -> Vec<u8> {
        self.game
            .legal_actions()
            .iter()
            .map(|&b| if b { 1 } else { 0 })
            .collect()
    }

    /// Helper method to create a JS result object.
    fn create_js_result(&self, result: StepResult) -> JsValue {
        let js_result = JsStepResult {
            board: self.game.board().to_vec(),
            score: self.game.score(),
            reward: result.reward,
            changed: result.changed,
            done: result.done,
        };
        serde_wasm_bindgen::to_value(&js_result).unwrap_or(JsValue::NULL)
    }
}

/// Initialize panic hook for better error messages in the browser console.
#[wasm_bindgen(start)]
pub fn init() {
    // Initialize solver lookup tables
    game_2048_core::solver::init_tables();
}

// =============================================================================
// Bitboard Solver Bindings
// =============================================================================

/// Solve the board and return the best move using the Rust bitboard solver.
///
/// Arguments:
/// - board_js: Array of 16 tile values (0 for empty, 2, 4, 8, ...)
/// - time_limit_ms: Time budget for search in milliseconds
///
/// Returns:
/// - Best action: 0=Up, 1=Down, 2=Left, 3=Right
#[wasm_bindgen(js_name = solveBoard)]
pub fn solve_board(board_js: Vec<u32>, time_limit_ms: u64) -> u8 {
    use game_2048_core::solver::{find_best_move, pack_board_from_tiles};

    // Convert JS board to bitboard
    let board = pack_board_from_tiles(&board_js);

    // Run solver
    let best_move = find_best_move(board, time_limit_ms);

    // Return action as u8
    best_move as u8
}

/// Test function to verify solver is working
#[wasm_bindgen(js_name = testSolver)]
pub fn test_solver() -> String {
    use game_2048_core::solver::{init_tables, pack_board_from_tiles, unpack_board_to_tiles};

    init_tables();

    let tiles = vec![
        2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 0, 0, 0, 0, 0,
    ];
    let board = pack_board_from_tiles(&tiles);
    let unpacked = unpack_board_to_tiles(board);

    format!("Solver ready! Test board: {:?}", unpacked)
}
