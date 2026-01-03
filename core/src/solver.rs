//! High-Performance Bitboard Solver for 2048
//!
//! This module implements an Expectimax algorithm using bitboards and pre-computed
//! lookup tables for maximum performance. The solver uses:
//! - Bitboard representation (u64 with 4 bits per tile)
//! - Pre-computed move tables (65,536 entries per direction)
//! - Snake gradient heuristic with 8-way symmetry
//! - Transposition table caching
//! - Iterative deepening with time budget

use std::collections::HashMap;
use std::time::{Duration, Instant};

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

/// Snake gradient weights (powers of 12 for strong preference)
const GRADIENT_POWERS: [f64; 16] = [
    68719476736.0,    // 12^15
    5726623061.0,     // 12^14
    477218588.0,      // 12^13
    39764832.0,       // 12^12
    429981696.0,      // 12^8
    5159780352.0,     // 12^9
    61917364224.0,    // 12^10
    743008370688.0,   // 12^11
    35831808.0,       // 12^7
    2985984.0,        // 12^6
    248832.0,         // 12^5
    20736.0,          // 12^4
    1.0,              // 12^0
    12.0,             // 12^1
    144.0,            // 12^2
    1728.0,           // 12^3
];

// =============================================================================
// Pre-computed Lookup Tables
// =============================================================================

/// Lookup table for moving a row left
/// Index: u16 representing 4 tiles (4 bits each)
/// Value: u16 representing result after move
static mut MOVE_LEFT_TABLE: [u16; 65536] = [0; 65536];
static mut MOVE_RIGHT_TABLE: [u16; 65536] = [0; 65536];
static mut ROW_SCORE_TABLE: [u16; 65536] = [0; 65536];

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
            for i in 0..4 {
                tiles[i] = ((row_u16 >> (i * 4)) & 0xF) as u8;
            }
            
            // Move left
            let (left_tiles, left_score) = compress_and_merge(&tiles);
            MOVE_LEFT_TABLE[row as usize] = pack_tiles(&left_tiles);
            ROW_SCORE_TABLE[row as usize] = left_score;
            
            // Move right (reverse, move left, reverse)
            let reversed = [tiles[3], tiles[2], tiles[1], tiles[0]];
            let (right_tiles, _) = compress_and_merge(&reversed);
            let right_result = [right_tiles[3], right_tiles[2], right_tiles[1], right_tiles[0]];
            MOVE_RIGHT_TABLE[row as usize] = pack_tiles(&right_result);
        }
        
        TABLES_INITIALIZED = true;
    }
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
            score += 1 << result[write_idx]; // 2^(tile+1)
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
    for i in 0..4 {
        packed |= (tiles[i] as u16) << (i * 4);
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

/// Get tile at position (row, col)
#[inline]
fn get_tile(board: Board, row: usize, col: usize) -> u8 {
    ((board >> ((row * 4 + col) * 4)) & 0xF) as u8
}

/// Set tile at position (row, col)
#[inline]
fn set_tile(board: Board, row: usize, col: usize, value: u8) -> Board {
    let pos = (row * 4 + col) * 4;
    let mask = !(0xFu64 << pos);
    (board & mask) | ((value as u64) << pos)
}

/// Apply a move to the board
pub fn apply_move(board: Board, action: Action) -> (Board, u32) {
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
// Symmetry Operations
// =============================================================================

/// Apply symmetry transformation (0-7)
/// 0: identity, 1: rotate 90°, 2: rotate 180°, 3: rotate 270°
/// 4: flip horizontal, 5: flip + rotate 90°, 6: flip + rotate 180°, 7: flip + rotate 270°
fn apply_symmetry(board: Board, symmetry: u8) -> Board {
    match symmetry {
        0 => board,
        1 => rotate_90(board),
        2 => rotate_180(board),
        3 => rotate_270(board),
        4 => flip_horizontal(board),
        5 => rotate_90(flip_horizontal(board)),
        6 => rotate_180(flip_horizontal(board)),
        7 => rotate_270(flip_horizontal(board)),
        _ => board,
    }
}

fn rotate_90(board: Board) -> Board {
    let mut result = 0u64;
    for row in 0..4 {
        for col in 0..4 {
            let tile = get_tile(board, row, col);
            let new_row = col;
            let new_col = 3 - row;
            result = set_tile(result, new_row, new_col, tile);
        }
    }
    result
}

fn rotate_180(board: Board) -> Board {
    let mut result = 0u64;
    for i in 0..16 {
        let tile = (board >> (i * 4)) & 0xF;
        result |= tile << ((15 - i) * 4);
    }
    result
}

fn rotate_270(board: Board) -> Board {
    rotate_90(rotate_180(board))
}

fn flip_horizontal(board: Board) -> Board {
    let mut result = 0u64;
    for row in 0..4 {
        for col in 0..4 {
            let tile = get_tile(board, row, col);
            result = set_tile(result, row, 3 - col, tile);
        }
    }
    result
}

// =============================================================================
// Evaluation
// =============================================================================

/// Evaluate board with gradient
fn evaluate_with_gradient(board: Board) -> f64 {
    let mut score = 0.0;
    for i in 0..16 {
        let tile = (board >> (i * 4)) & 0xF;
        if tile > 0 {
            score += GRADIENT_POWERS[i] * (1u64 << tile) as f64;
        }
    }
    score
}

/// Check if max tile is in corner (position 0)
fn max_tile_in_corner(board: Board) -> bool {
    let corner_tile = board & 0xF;
    if corner_tile == 0 {
        return false;
    }
    
    for i in 1..16 {
        let tile = (board >> (i * 4)) & 0xF;
        if tile > corner_tile {
            return false;
        }
    }
    true
}

/// Evaluate with 8-way symmetry
fn evaluate(board: Board) -> f64 {
    let mut best_score = f64::MIN;
    
    for sym in 0..8 {
        let transformed = apply_symmetry(board, sym);
        let score = evaluate_with_gradient(transformed);
        best_score = best_score.max(score);
    }
    
    // Corner penalty
    if !max_tile_in_corner(board) {
        best_score *= 0.2;
    }
    
    best_score
}

// =============================================================================
// Expectimax Solver
// =============================================================================

pub struct Solver {
    transposition_table: HashMap<Board, f64>,
    nodes_searched: usize,
    start_time: Instant,
}

impl Solver {
    pub fn new() -> Self {
        Self {
            transposition_table: HashMap::new(),
            nodes_searched: 0,
            start_time: Instant::now(),
        }
    }
    
    fn expectimax(
        &mut self,
        board: Board,
        depth: usize,
        is_max_node: bool,
        time_limit: Duration,
    ) -> f64 {
        self.nodes_searched += 1;
        
        // Time check
        if self.start_time.elapsed() > time_limit {
            return evaluate(board);
        }
        
        // Base case
        if depth == 0 {
            return evaluate(board);
        }
        
        // Transposition table lookup
        if let Some(&cached) = self.transposition_table.get(&board) {
            return cached;
        }
        
        let score = if is_max_node {
            self.max_node(board, depth, time_limit)
        } else {
            self.chance_node(board, depth, time_limit)
        };
        
        // Cache (limit size to avoid memory bloat)
        if self.transposition_table.len() < 100_000 {
            self.transposition_table.insert(board, score);
        }
        
        score
    }
    
    fn max_node(&mut self, board: Board, depth: usize, time_limit: Duration) -> f64 {
        let mut best = f64::MIN;
        let mut has_move = false;
        
        for action in [Action::Up, Action::Down, Action::Left, Action::Right] {
            let (new_board, move_score) = apply_move(board, action);
            if new_board == board {
                continue; // No change
            }
            has_move = true;
            
            let eval = self.expectimax(new_board, depth - 1, false, time_limit)
                + move_score as f64 * 0.1;
            
            best = best.max(eval);
        }
        
        if has_move { best } else { evaluate(board) }
    }
    
    fn chance_node(&mut self, board: Board, depth: usize, time_limit: Duration) -> f64 {
        let empties = get_empty_positions(board);
        if empties.is_empty() {
            return evaluate(board);
        }
        
        let mut total = 0.0;
        
        for &pos in &empties {
            // Spawn 2 (90%)
            let board_with_2 = board | (1u64 << (pos * 4));
            total += 0.9 * self.expectimax(board_with_2, depth - 1, true, time_limit);
            
            // Spawn 4 (10%)
            let board_with_4 = board | (2u64 << (pos * 4));
            total += 0.1 * self.expectimax(board_with_4, depth - 1, true, time_limit);
        }
        
        total / empties.len() as f64
    }
}

// =============================================================================
// Main Interface
// =============================================================================

/// Find the best move using iterative deepening
pub fn find_best_move(board: Board, time_limit_ms: u64) -> Action {
    let time_limit = Duration::from_millis(time_limit_ms);
    let mut best_action = Action::Left;
    
    // Ensure tables are initialized
    init_tables();
    
    // Iterative deepening: 2, 4, 6, 8, ...
    for depth in (2..=12).step_by(2) {
        let mut solver = Solver::new();
        
        if solver.start_time.elapsed() > time_limit * 80 / 100 {
            break;
        }
        
        let mut depth_best = Action::Left;
        let mut depth_best_score = f64::MIN;
        
        for action in [Action::Up, Action::Down, Action::Left, Action::Right] {
            let (new_board, score) = apply_move(board, action);
            if new_board == board {
                continue;
            }
            
            let eval = solver.expectimax(new_board, depth - 1, false, time_limit)
                + score as f64 * 0.1;
            
            if eval > depth_best_score {
                depth_best_score = eval;
                depth_best = action;
            }
        }
        
        best_action = depth_best;
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
        let tiles = vec![2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 0, 0, 0, 0, 0];
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
}
