//! High-Performance Bitboard Solver for 2048
//!
//! This module implements an Expectimax algorithm using bitboards and pre-computed
//! lookup tables for maximum performance. The solver uses:
//! - Bitboard representation (u64 with 4 bits per tile)
//! - Pre-computed move tables (65,536 entries per direction)
//! - Per-row precomputed heuristic (empties, merges, monotonicity, sum) — the
//!   well-established "nneonneo" evaluation that reliably reaches the 2048 tile
//! - Transposition table caching keyed on (board, depth)
//! - Adaptive search depth based on board complexity, with a time budget

use instant::{Duration, Instant};
use std::collections::HashMap;

// =============================================================================
// Types and Constants
// =============================================================================

/// Board representation: 64 bits, 4 bits per tile (16 tiles)
/// Each 4-bit value represents the power of 2 (0 = empty, 1 = 2, 2 = 4, etc.)
pub type Board = u64;

/// Action/Move direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Action {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

impl Action {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Action::Up),
            1 => Some(Action::Down),
            2 => Some(Action::Left),
            3 => Some(Action::Right),
            _ => None,
        }
    }
}

// Heuristic weights (nneonneo-style). The board score is the sum of a
// precomputed per-row heuristic over the four rows and the four columns.
const HEUR_LOST_PENALTY: f64 = 200000.0;
const HEUR_MONOTONICITY_POWER: f64 = 4.0;
const HEUR_MONOTONICITY_WEIGHT: f64 = 47.0;
const HEUR_SUM_POWER: f64 = 3.5;
const HEUR_SUM_WEIGHT: f64 = 11.0;
const HEUR_MERGES_WEIGHT: f64 = 700.0;
const HEUR_EMPTY_WEIGHT: f64 = 270.0;

// =============================================================================
// Pre-computed Lookup Tables
// =============================================================================

/// Lookup table for moving a row left
/// Index: u16 representing 4 tiles (4 bits each)
/// Value: u16 representing result after move
static mut MOVE_LEFT_TABLE: [u16; 65536] = [0; 65536];
static mut MOVE_RIGHT_TABLE: [u16; 65536] = [0; 65536];
static mut ROW_SCORE_TABLE: [u16; 65536] = [0; 65536];

/// Precomputed per-row heuristic value (empties, merges, monotonicity, sum).
/// The board heuristic is the sum over the four rows plus the four columns.
static mut HEUR_TABLE: [f64; 65536] = [0.0; 65536];

static mut TABLES_INITIALIZED: bool = false;

/// Initialize lookup tables (call once at startup)
pub fn init_tables() {
    unsafe {
        if TABLES_INITIALIZED {
            return;
        }

        for row in 0..65536u32 {
            let row_u16 = row as u16;

            // Extract tiles
            let mut tiles = [0u8; 4];
            for (i, tile) in tiles.iter_mut().enumerate() {
                *tile = ((row_u16 >> (i * 4)) & 0xF) as u8;
            }

            // Move left
            let (left_tiles, left_score) = compress_and_merge(&tiles);
            MOVE_LEFT_TABLE[row as usize] = pack_tiles(&left_tiles);
            ROW_SCORE_TABLE[row as usize] = left_score;

            // Move right (reverse, move left, reverse)
            let reversed = [tiles[3], tiles[2], tiles[1], tiles[0]];
            let (right_tiles, _) = compress_and_merge(&reversed);
            let right_result = [
                right_tiles[3],
                right_tiles[2],
                right_tiles[1],
                right_tiles[0],
            ];
            MOVE_RIGHT_TABLE[row as usize] = pack_tiles(&right_result);

            // Per-row heuristic value
            HEUR_TABLE[row as usize] = row_heuristic(&tiles);
        }

        TABLES_INITIALIZED = true;
    }
}

/// Compute the heuristic contribution of a single row of tile exponents.
fn row_heuristic(tiles: &[u8; 4]) -> f64 {
    // Empty cells and merge potential.
    let mut empties = 0i32;
    let mut merges = 0i32;
    let mut prev = 0u8;
    let mut counter = 0i32;
    let mut sum = 0.0;

    for &t in tiles {
        sum += (t as f64).powf(HEUR_SUM_POWER);
        if t == 0 {
            empties += 1;
        } else {
            if prev == t {
                counter += 1;
            } else if counter > 0 {
                merges += 1 + counter;
                counter = 0;
            }
            prev = t;
        }
    }
    if counter > 0 {
        merges += 1 + counter;
    }

    // Monotonicity: prefer rows that are strictly increasing or decreasing.
    let mut mono_left = 0.0;
    let mut mono_right = 0.0;
    for i in 0..3 {
        let a = (tiles[i] as f64).powf(HEUR_MONOTONICITY_POWER);
        let b = (tiles[i + 1] as f64).powf(HEUR_MONOTONICITY_POWER);
        if a > b {
            mono_left += a - b;
        } else {
            mono_right += b - a;
        }
    }

    HEUR_LOST_PENALTY + HEUR_EMPTY_WEIGHT * empties as f64 + HEUR_MERGES_WEIGHT * merges as f64
        - HEUR_MONOTONICITY_WEIGHT * mono_left.min(mono_right)
        - HEUR_SUM_WEIGHT * sum
}

/// Compress and merge a single row (moving towards index 0)
fn compress_and_merge(tiles: &[u8; 4]) -> ([u8; 4], u16) {
    let mut result = [0u8; 4];
    let mut score = 0u16;
    let mut write_idx = 0;

    // First pass: compress (remove zeros)
    let mut temp = [0u8; 4];
    let mut temp_len = 0;
    for &tile in tiles {
        if tile != 0 {
            temp[temp_len] = tile;
            temp_len += 1;
        }
    }

    // Second pass: merge
    let mut i = 0;
    while i < temp_len {
        if i + 1 < temp_len && temp[i] == temp[i + 1] && temp[i] != 0 {
            // Merge
            result[write_idx] = temp[i] + 1;
            // Use saturating arithmetic to avoid overflow in debug builds
            score = score.saturating_add(1u16 << result[write_idx].min(15));
            write_idx += 1;
            i += 2;
        } else {
            // No merge
            result[write_idx] = temp[i];
            write_idx += 1;
            i += 1;
        }
    }

    (result, score)
}

/// Pack 4 tiles into a u16
fn pack_tiles(tiles: &[u8; 4]) -> u16 {
    let mut packed = 0u16;
    for (i, &tile) in tiles.iter().enumerate() {
        packed |= (tile as u16) << (i * 4);
    }
    packed
}

// =============================================================================
// Bitboard Operations
// =============================================================================

/// Extract a row from the board
#[inline]
fn get_row(board: Board, row: usize) -> u16 {
    ((board >> (row * 16)) & 0xFFFF) as u16
}

/// Set a row in the board
#[inline]
fn set_row(board: Board, row: usize, row_data: u16) -> Board {
    let mask = !(0xFFFFu64 << (row * 16));
    (board & mask) | ((row_data as u64) << (row * 16))
}

/// Transpose the board (swap rows and columns)
fn transpose(board: Board) -> Board {
    let mut result = 0u64;
    for row in 0..4 {
        for col in 0..4 {
            let tile = (board >> ((row * 4 + col) * 4)) & 0xF;
            result |= tile << ((col * 4 + row) * 4);
        }
    }
    result
}

/// Apply a move to the board
pub fn apply_move(board: Board, action: Action) -> (Board, u32) {
    // Ensure tables are initialized before use
    init_tables();

    unsafe {
        match action {
            Action::Left => {
                let mut new_board = 0u64;
                let mut score = 0u32;
                for row in 0..4 {
                    let row_data = get_row(board, row);
                    new_board = set_row(new_board, row, MOVE_LEFT_TABLE[row_data as usize]);
                    score += ROW_SCORE_TABLE[row_data as usize] as u32;
                }
                (new_board, score)
            }
            Action::Right => {
                let mut new_board = 0u64;
                let mut score = 0u32;
                for row in 0..4 {
                    let row_data = get_row(board, row);
                    new_board = set_row(new_board, row, MOVE_RIGHT_TABLE[row_data as usize]);
                    score += ROW_SCORE_TABLE[row_data as usize] as u32;
                }
                (new_board, score)
            }
            Action::Up => {
                let transposed = transpose(board);
                let (result, score) = apply_move(transposed, Action::Left);
                (transpose(result), score)
            }
            Action::Down => {
                let transposed = transpose(board);
                let (result, score) = apply_move(transposed, Action::Right);
                (transpose(result), score)
            }
        }
    }
}

/// Get empty positions
fn get_empty_positions(board: Board) -> Vec<usize> {
    let mut positions = Vec::with_capacity(16);
    for i in 0..16 {
        if (board >> (i * 4)) & 0xF == 0 {
            positions.push(i);
        }
    }
    positions
}

// =============================================================================
// Evaluation
// =============================================================================

/// Evaluate a board as the sum of the per-row heuristic over the four rows and
/// the four columns (rows of the transposed board). The heuristic favours empty
/// cells, available merges, and monotone arrangements that keep large tiles
/// packed into a corner — the key to reliably reaching the 2048 tile.
fn evaluate(board: Board) -> f64 {
    let transposed = transpose(board);
    let mut score = 0.0;
    unsafe {
        for row in 0..4 {
            score += HEUR_TABLE[get_row(board, row) as usize];
            score += HEUR_TABLE[get_row(transposed, row) as usize];
        }
    }
    score
}

// =============================================================================
// Expectimax Solver
// =============================================================================

/// Probability below which a chance-node branch is cut off (its subtree
/// becomes too unlikely to be worth expanding). Lets the search go deeper.
const CPROB_THRESH: f64 = 0.0001;

pub struct Solver {
    /// Cache of evaluated chance nodes, keyed on (board, remaining depth).
    transposition_table: HashMap<(Board, usize), f64>,
    nodes_searched: usize,
    start_time: Instant,
    time_limit: Duration,
}

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver {
    pub fn new() -> Self {
        Self {
            transposition_table: HashMap::new(),
            nodes_searched: 0,
            start_time: Instant::now(),
            time_limit: Duration::from_millis(100),
        }
    }

    fn expectimax(&mut self, board: Board, depth: usize, is_max_node: bool, prob: f64) -> f64 {
        self.nodes_searched += 1;

        // Cut off unlikely branches, exhausted depth, or expired time budget.
        if depth == 0 || prob < CPROB_THRESH || self.start_time.elapsed() > self.time_limit {
            return evaluate(board);
        }

        if is_max_node {
            self.max_node(board, depth, prob)
        } else {
            self.chance_node(board, depth, prob)
        }
    }

    fn max_node(&mut self, board: Board, depth: usize, prob: f64) -> f64 {
        let mut best = f64::MIN;
        let mut has_move = false;

        for action in [Action::Up, Action::Down, Action::Left, Action::Right] {
            let (new_board, _move_score) = apply_move(board, action);
            if new_board == board {
                continue; // No change
            }
            has_move = true;
            best = best.max(self.expectimax(new_board, depth - 1, false, prob));
        }

        if has_move {
            best
        } else {
            // Game over: no legal moves. The packed board has no empties, so its
            // heuristic is already poor relative to live positions.
            evaluate(board)
        }
    }

    fn chance_node(&mut self, board: Board, depth: usize, prob: f64) -> f64 {
        // Reuse a previously computed value for this exact (board, depth).
        if let Some(&cached) = self.transposition_table.get(&(board, depth)) {
            return cached;
        }

        let empties = get_empty_positions(board);
        if empties.is_empty() {
            return evaluate(board);
        }

        let n = empties.len() as f64;
        let prob_each = prob / n;
        let mut total = 0.0;

        for &pos in &empties {
            // Spawn 2 (90%)
            let board_with_2 = board | (1u64 << (pos * 4));
            total += 0.9 * self.expectimax(board_with_2, depth - 1, true, prob_each * 0.9);

            // Spawn 4 (10%)
            let board_with_4 = board | (2u64 << (pos * 4));
            total += 0.1 * self.expectimax(board_with_4, depth - 1, true, prob_each * 0.1);
        }

        let result = total / n;

        // Cache (limit size to avoid memory bloat)
        if self.transposition_table.len() < 100_000 {
            self.transposition_table.insert((board, depth), result);
        }

        result
    }
}

/// Count distinct non-empty tile values on the board.
fn count_distinct_tiles(board: Board) -> u32 {
    let mut seen = 0u16; // bitmask over the 16 possible tile exponents
    for i in 0..16 {
        let tile = ((board >> (i * 4)) & 0xF) as u8;
        if tile != 0 {
            seen |= 1 << tile;
        }
    }
    seen.count_ones()
}

// =============================================================================
// Main Interface
// =============================================================================

/// Find the best move via depth-limited expectimax.
///
/// Search depth adapts to board complexity (deeper when more distinct tiles are
/// present, which is when planning matters most). `time_limit_ms` is a hard
/// safety cap: if the search runs long, remaining nodes fall back to the
/// heuristic so a move is always returned promptly.
pub fn find_best_move(board: Board, time_limit_ms: u64) -> Action {
    // Ensure tables are initialized
    init_tables();

    let mut solver = Solver::new();
    solver.start_time = Instant::now();
    solver.time_limit = Duration::from_millis(time_limit_ms);

    // Adaptive depth: max(3, distinct_tiles - 2).
    let depth = (count_distinct_tiles(board) as i32 - 2).max(3) as usize;

    let mut best_action = Action::Left;
    let mut best_score = f64::MIN;

    for action in [Action::Up, Action::Down, Action::Left, Action::Right] {
        let (new_board, _score) = apply_move(board, action);
        if new_board == board {
            continue;
        }

        // Resulting board is a chance node (a random tile spawns next).
        let eval = solver.expectimax(new_board, depth, false, 1.0);
        if eval > best_score {
            best_score = eval;
            best_action = action;
        }
    }

    best_action
}

/// Convert from Vec<u32> (tile values) to Board (bitboard)
pub fn pack_board_from_tiles(tiles: &[u32]) -> Board {
    let mut board = 0u64;
    for (i, &tile) in tiles.iter().enumerate().take(16) {
        let power = if tile == 0 {
            0
        } else {
            // Convert tile value to power of 2
            (tile as f64).log2() as u8
        };
        board |= (power as u64) << (i * 4);
    }
    board
}

/// Convert Board (bitboard) to Vec<u32> (tile values)
pub fn unpack_board_to_tiles(board: Board) -> Vec<u32> {
    let mut tiles = Vec::with_capacity(16);
    for i in 0..16 {
        let power = ((board >> (i * 4)) & 0xF) as u8;
        let tile = if power == 0 { 0 } else { 1 << power };
        tiles.push(tile);
    }
    tiles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tables_init() {
        init_tables();
        unsafe {
            assert!(TABLES_INITIALIZED);
        }
    }

    #[test]
    fn test_pack_unpack() {
        let tiles = vec![
            2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 0, 0, 0, 0, 0,
        ];
        let board = pack_board_from_tiles(&tiles);
        let unpacked = unpack_board_to_tiles(board);
        assert_eq!(tiles, unpacked);
    }

    #[test]
    fn test_move_left() {
        init_tables();
        // [2, 2, 4, 0] should become [4, 4, 0, 0]
        let tiles = vec![2, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let board = pack_board_from_tiles(&tiles);
        let (new_board, score) = apply_move(board, Action::Left);
        let result = unpack_board_to_tiles(new_board);
        assert_eq!(result[0], 4);
        assert_eq!(result[1], 4);
        assert_eq!(score, 4);
    }

    #[test]
    fn test_count_distinct_tiles() {
        let tiles = vec![2, 4, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let board = pack_board_from_tiles(&tiles);
        // distinct non-empty values: 2, 4, 8 => 3
        assert_eq!(count_distinct_tiles(board), 3);
    }

    #[test]
    fn test_find_best_move_returns_legal_move() {
        init_tables();
        // A mid-game board with several legal moves available.
        let tiles = vec![2, 4, 8, 16, 0, 4, 8, 16, 0, 0, 8, 32, 0, 0, 0, 2];
        let board = pack_board_from_tiles(&tiles);
        let action = find_best_move(board, 100);
        // The chosen move must actually change the board (i.e. be legal).
        let (new_board, _) = apply_move(board, action);
        assert_ne!(new_board, board, "solver returned a no-op move");
    }

    /// End-to-end self-play check: the solver should reliably reach the 2048
    /// tile. Ignored by default because it plays full games (slow); run with
    /// `cargo test -- --ignored solver_reaches_2048`.
    #[test]
    #[ignore]
    fn solver_reaches_2048() {
        use crate::Game;

        let games = 20;
        let mut wins = 0;
        for seed in 0..games {
            let mut game = Game::new(seed);
            while !game.is_done() {
                let tiles: Vec<u32> = game.board().iter().map(|&t| t as u32).collect();
                let board = pack_board_from_tiles(&tiles);
                let action = find_best_move(board, 100);
                game.step(action_to_engine(action));
                if game.board().iter().any(|&t| t >= 2048) {
                    wins += 1;
                    break;
                }
            }
        }
        // Expect a strong win rate; allow some slack for unlucky seeds.
        assert!(
            wins as f64 / games as f64 >= 0.8,
            "only reached 2048 in {wins}/{games} games"
        );
    }

    fn action_to_engine(action: Action) -> crate::Action {
        match action {
            Action::Up => crate::Action::Up,
            Action::Down => crate::Action::Down,
            Action::Left => crate::Action::Left,
            Action::Right => crate::Action::Right,
        }
    }
}
