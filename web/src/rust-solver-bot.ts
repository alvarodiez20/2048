/**
 * Rust Bitboard Solver Bot
 * 
 * High-performance Expectimax solver implemented in Rust using bitboards.
 * This is 10-30x faster than the JavaScript Expectimax implementation.
 */

import { solveBoard } from './wasm-pkg/game_2048_wasm.js';

export type Action = 0 | 1 | 2 | 3; // Up, Down, Left, Right

export class RustSolverBot {
    private timeLimitMs: number;

    constructor(timeLimitMs: number = 100) {
        this.timeLimitMs = timeLimitMs;
    }

    /**
     * Get the best action for the current board state.
     */
    getAction(board: number[], _legalActions: boolean[]): Action {
        // Convert to Uint32Array for WASM
        const boardArray = new Uint32Array(board);

        // Call Rust solver (solveBoard expects BigInt for time limit)
        const action = solveBoard(boardArray, BigInt(this.timeLimitMs)) as Action;

        return action;
    }

    /**
     * Async version for compatibility with AI player interface.
     */
    async getActionAsync(board: number[], _legalActions: boolean[]): Promise<Action> {
        return this.getAction(board, _legalActions);
    }

    // Interface compatibility methods
    isLoaded(): boolean { return true; }
    isLoading(): boolean { return false; }
    async load(): Promise<void> { }
    getName(): string { return 'Rust Solver (Bitboard)'; }
    getType(): 'rust-solver' { return 'rust-solver'; }
}

export const ACTION_NAMES = ['↑ Up', '↓ Down', '← Left', '→ Right'];

export function getActionName(action: Action): string {
    return ACTION_NAMES[action];
}
