/**
 * Expectimax Bot for 2048
 * 
 * A corner-locked, risk-averse expectimax AI that uses direct WASM integration
 * (no DOM parsing, no synthetic keyboard events).
 * 
 * Features:
 * - Expectimax search with iterative deepening
 * - Risk-averse chance nodes: (1-α)*avg + α*min
 * - Corner-locked TOP-LEFT heuristic
 * - Time-budgeted search
 * - State-change verification to prevent stuck loops
 */

// =============================================================================
// Configuration (Tunable Parameters)
// =============================================================================

export const CONFIG = {
    TIME_LIMIT_MS: 400,        // Per-move time budget
    ALPHA: 0.25,               // Risk aversion (0 = neutral, 1 = very pessimistic)
    MAX_DEPTH: 8,              // Maximum search depth
    MAX_EMPTIES_FULL: 6,       // Max empties to fully enumerate in chance nodes
};

// Heuristic weights
const WEIGHTS = {
    EMPTY_BONUS: 270,
    CORNER_BONUS: 50000,
    MONOTONICITY: 47,
    SMOOTHNESS: -10,
    MERGE_POTENTIAL: 700,
    MOBILITY: 200,
    DEATH_PENALTY: -100000,
};

// Snake pattern weights (anchored to TOP-LEFT corner)
// Higher values in top-left encourage keeping max tile there
const SNAKE_WEIGHTS = new Float64Array([
    Math.pow(4, 15), Math.pow(4, 14), Math.pow(4, 13), Math.pow(4, 12),
    Math.pow(4, 8), Math.pow(4, 9), Math.pow(4, 10), Math.pow(4, 11),
    Math.pow(4, 7), Math.pow(4, 6), Math.pow(4, 5), Math.pow(4, 4),
    Math.pow(4, 0), Math.pow(4, 1), Math.pow(4, 2), Math.pow(4, 3),
]);

// =============================================================================
// Types
// =============================================================================

export type Board = Uint16Array;
export type Action = 0 | 1 | 2 | 3; // Up, Down, Left, Right

export interface SearchResult {
    bestAction: Action;
    score: number;
    nodesSearched: number;
    depthReached: number;
    timeMs: number;
    empties: number;
}

export interface MoveResult {
    board: Board;
    changed: boolean;
    score: number;
}

export interface BotStats {
    gamesPlayed: number;
    maxTileReached: number;
    totalScore: number;
    reached512: number;
    reached1024: number;
    reached2048: number;
    reached4096: number;
}

// =============================================================================
// Board Utilities
// =============================================================================

function boardHash(board: Board): string {
    return Array.from(board).join(',');
}

function countEmpties(board: Board): number {
    let count = 0;
    for (let i = 0; i < 16; i++) {
        if (board[i] === 0) count++;
    }
    return count;
}

function getEmptyIndices(board: Board): number[] {
    const empties: number[] = [];
    for (let i = 0; i < 16; i++) {
        if (board[i] === 0) empties.push(i);
    }
    return empties;
}

function getMaxTile(board: Board): number {
    let max = 0;
    for (let i = 0; i < 16; i++) {
        if (board[i] > max) max = board[i];
    }
    return max;
}

function toLog2(value: number): number {
    return value === 0 ? 0 : Math.log2(value);
}

// =============================================================================
// Move Simulation (Fast, Pure JS)
// =============================================================================

function compressAndMerge(line: Uint16Array): number {
    let score = 0;

    // Compress: move non-zeros to front
    let writeIdx = 0;
    for (let i = 0; i < 4; i++) {
        if (line[i] !== 0) {
            line[writeIdx++] = line[i];
        }
    }
    while (writeIdx < 4) line[writeIdx++] = 0;

    // Merge adjacent equals
    for (let i = 0; i < 3; i++) {
        if (line[i] !== 0 && line[i] === line[i + 1]) {
            line[i] *= 2;
            score += line[i];
            line[i + 1] = 0;
        }
    }

    // Compress again
    writeIdx = 0;
    for (let i = 0; i < 4; i++) {
        if (line[i] !== 0) {
            line[writeIdx++] = line[i];
        }
    }
    while (writeIdx < 4) line[writeIdx++] = 0;

    return score;
}

function applyMove(board: Board, action: Action): MoveResult {
    const newBoard = new Uint16Array(board);
    let totalScore = 0;
    let changed = false;

    const line = new Uint16Array(4);

    if (action === 0) {
        // Up: process columns top to bottom
        for (let col = 0; col < 4; col++) {
            for (let row = 0; row < 4; row++) line[row] = newBoard[row * 4 + col];
            const oldLine = Uint16Array.from(line);
            totalScore += compressAndMerge(line);
            for (let row = 0; row < 4; row++) newBoard[row * 4 + col] = line[row];
            if (!changed) {
                for (let row = 0; row < 4; row++) {
                    if (oldLine[row] !== line[row]) { changed = true; break; }
                }
            }
        }
    } else if (action === 1) {
        // Down: process columns bottom to top
        for (let col = 0; col < 4; col++) {
            for (let row = 0; row < 4; row++) line[row] = newBoard[(3 - row) * 4 + col];
            const oldLine = Uint16Array.from(line);
            totalScore += compressAndMerge(line);
            for (let row = 0; row < 4; row++) newBoard[(3 - row) * 4 + col] = line[row];
            if (!changed) {
                for (let row = 0; row < 4; row++) {
                    if (oldLine[row] !== line[row]) { changed = true; break; }
                }
            }
        }
    } else if (action === 2) {
        // Left: process rows left to right
        for (let row = 0; row < 4; row++) {
            for (let col = 0; col < 4; col++) line[col] = newBoard[row * 4 + col];
            const oldLine = Uint16Array.from(line);
            totalScore += compressAndMerge(line);
            for (let col = 0; col < 4; col++) newBoard[row * 4 + col] = line[col];
            if (!changed) {
                for (let col = 0; col < 4; col++) {
                    if (oldLine[col] !== line[col]) { changed = true; break; }
                }
            }
        }
    } else {
        // Right: process rows right to left
        for (let row = 0; row < 4; row++) {
            for (let col = 0; col < 4; col++) line[col] = newBoard[row * 4 + (3 - col)];
            const oldLine = Uint16Array.from(line);
            totalScore += compressAndMerge(line);
            for (let col = 0; col < 4; col++) newBoard[row * 4 + (3 - col)] = line[col];
            if (!changed) {
                for (let col = 0; col < 4; col++) {
                    if (oldLine[col] !== line[col]) { changed = true; break; }
                }
            }
        }
    }

    return { board: newBoard, changed, score: totalScore };
}

function getLegalMoves(board: Board): Action[] {
    const legal: Action[] = [];
    for (const action of [0, 1, 2, 3] as Action[]) {
        if (applyMove(board, action).changed) {
            legal.push(action);
        }
    }
    return legal;
}

// =============================================================================
// Heuristic Evaluation (Corner-Locked TOP-LEFT)
// =============================================================================

function monotonicity(board: Board): number {
    let score = 0;

    // Rows
    for (let row = 0; row < 4; row++) {
        let inc = 0, dec = 0;
        for (let col = 0; col < 3; col++) {
            const curr = toLog2(board[row * 4 + col]);
            const next = toLog2(board[row * 4 + col + 1]);
            if (curr > next) dec += curr - next;
            else inc += next - curr;
        }
        score += Math.max(inc, dec);
    }

    // Columns
    for (let col = 0; col < 4; col++) {
        let inc = 0, dec = 0;
        for (let row = 0; row < 3; row++) {
            const curr = toLog2(board[row * 4 + col]);
            const next = toLog2(board[(row + 1) * 4 + col]);
            if (curr > next) dec += curr - next;
            else inc += next - curr;
        }
        score += Math.max(inc, dec);
    }

    return score;
}

function smoothness(board: Board): number {
    let penalty = 0;

    for (let row = 0; row < 4; row++) {
        for (let col = 0; col < 4; col++) {
            const val = toLog2(board[row * 4 + col]);
            if (val === 0) continue;

            // Right neighbor
            if (col < 3) {
                const right = toLog2(board[row * 4 + col + 1]);
                if (right > 0) penalty += Math.abs(val - right);
            }
            // Down neighbor
            if (row < 3) {
                const down = toLog2(board[(row + 1) * 4 + col]);
                if (down > 0) penalty += Math.abs(val - down);
            }
        }
    }

    return penalty;
}

function mergePotential(board: Board): number {
    let count = 0;

    for (let row = 0; row < 4; row++) {
        for (let col = 0; col < 4; col++) {
            const val = board[row * 4 + col];
            if (val === 0) continue;

            // Check right neighbor
            if (col < 3 && board[row * 4 + col + 1] === val) count++;
            // Check down neighbor
            if (row < 3 && board[(row + 1) * 4 + col] === val) count++;
        }
    }

    return count;
}

function evaluate(board: Board): number {
    let score = 0;

    // 1. Snake/corner weights (commit to TOP-LEFT)
    for (let i = 0; i < 16; i++) {
        if (board[i] > 0) {
            score += toLog2(board[i]) * SNAKE_WEIGHTS[i];
        }
    }

    // 2. Empty cell bonus (larger when board is tight)
    const empties = countEmpties(board);
    score += WEIGHTS.EMPTY_BONUS * empties * empties;

    // 3. Monotonicity
    score += WEIGHTS.MONOTONICITY * monotonicity(board);

    // 4. Smoothness penalty
    score += WEIGHTS.SMOOTHNESS * smoothness(board);

    // 5. Merge potential
    score += WEIGHTS.MERGE_POTENTIAL * mergePotential(board);

    // 6. Mobility
    const mobility = getLegalMoves(board).length;
    score += WEIGHTS.MOBILITY * mobility;

    // 7. Death penalty
    if (mobility <= 1) {
        score += WEIGHTS.DEATH_PENALTY * (2 - mobility);
    }

    // 8. Max tile in TOP-LEFT corner bonus
    const maxTile = getMaxTile(board);
    if (board[0] === maxTile && maxTile > 0) {
        score += WEIGHTS.CORNER_BONUS * toLog2(maxTile);
    }

    return score;
}

// =============================================================================
// Expectimax Search
// =============================================================================

let nodesSearched = 0;

function selectImportantEmpties(board: Board, empties: number[], maxCount: number): number[] {
    if (empties.length <= maxCount) return empties;

    // Score empties by adjacency to high tiles (deterministic selection)
    const scored = empties.map(idx => {
        let adjacencyScore = 0;
        const row = Math.floor(idx / 4);
        const col = idx % 4;

        const neighbors = [
            row > 0 ? (row - 1) * 4 + col : -1,
            row < 3 ? (row + 1) * 4 + col : -1,
            col > 0 ? row * 4 + (col - 1) : -1,
            col < 3 ? row * 4 + (col + 1) : -1,
        ];

        for (const n of neighbors) {
            if (n >= 0 && board[n] > 0) {
                adjacencyScore += toLog2(board[n]);
            }
        }

        return { idx, score: adjacencyScore };
    });

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, maxCount).map(s => s.idx);
}

function expectimaxSearch(
    board: Board,
    depth: number,
    isMaxNode: boolean,
    alpha: number,
    startTime: number,
    timeLimitMs: number,
    transTable: Map<string, number>
): number {
    nodesSearched++;

    // Time check
    if (performance.now() - startTime > timeLimitMs) {
        return evaluate(board);
    }

    // Base case
    if (depth <= 0) {
        return evaluate(board);
    }

    const key = `${boardHash(board)}|${depth}|${isMaxNode ? 1 : 0}`;
    const cached = transTable.get(key);
    if (cached !== undefined) {
        return cached;
    }

    let result: number;

    if (isMaxNode) {
        // Player's turn: maximize over legal moves
        let bestScore = -Infinity;
        let hasMove = false;

        for (const action of [0, 1, 2, 3] as Action[]) {
            const moveResult = applyMove(board, action);
            if (!moveResult.changed) continue;
            hasMove = true;

            const evalScore = expectimaxSearch(
                moveResult.board,
                depth - 1,
                false,
                alpha,
                startTime,
                timeLimitMs,
                transTable
            ) + moveResult.score * 0.1;

            if (evalScore > bestScore) {
                bestScore = evalScore;
            }
        }

        result = hasMove ? bestScore : evaluate(board);
    } else {
        // Chance node: spawn tiles (risk-averse blend)
        const empties = getEmptyIndices(board);
        if (empties.length === 0) {
            result = evaluate(board);
        } else {
            const selectedEmpties = selectImportantEmpties(
                board, empties, CONFIG.MAX_EMPTIES_FULL
            );

            const scores: number[] = [];

            for (const idx of selectedEmpties) {
                // Spawn 2 (90%) and 4 (10%)
                for (const [tile, prob] of [[2, 0.9], [4, 0.1]] as const) {
                    const newBoard = new Uint16Array(board);
                    newBoard[idx] = tile;

                    const evalScore = expectimaxSearch(
                        newBoard,
                        depth - 1,
                        true,
                        alpha,
                        startTime,
                        timeLimitMs,
                        transTable
                    ) * prob;

                    scores.push(evalScore);
                }
            }

            // Risk-averse: (1 - alpha) * avg + alpha * min
            const sum = scores.reduce((a, b) => a + b, 0);
            const avg = sum / scores.length;
            const min = Math.min(...scores);
            result = (1 - alpha) * avg + alpha * min;
        }
    }

    // Cache result (limit cache size)
    if (transTable.size < 100000) {
        transTable.set(key, result);
    }

    return result;
}

// =============================================================================
// Main Search Interface
// =============================================================================

export function findBestMove(board: Board, timeLimitMs: number = CONFIG.TIME_LIMIT_MS): SearchResult {
    const startTime = performance.now();
    let bestAction: Action = 2; // Default to Left
    let bestScore = -Infinity;
    let depthReached = 0;
    nodesSearched = 0;

    // Get legal moves first
    const legalMoves = getLegalMoves(board);
    if (legalMoves.length === 0) {
        return {
            bestAction: 0,
            score: evaluate(board),
            nodesSearched: 1,
            depthReached: 0,
            timeMs: performance.now() - startTime,
            empties: countEmpties(board),
        };
    }

    if (legalMoves.length === 1) {
        return {
            bestAction: legalMoves[0],
            score: evaluate(board),
            nodesSearched: 1,
            depthReached: 0,
            timeMs: performance.now() - startTime,
            empties: countEmpties(board),
        };
    }

    // Iterative deepening
    for (let depth = 2; depth <= CONFIG.MAX_DEPTH; depth++) {
        const transTable = new Map<string, number>();
        let depthBestAction: Action = bestAction;
        let depthBestScore = -Infinity;

        for (const action of legalMoves) {
            const moveResult = applyMove(board, action);

            const evalScore = expectimaxSearch(
                moveResult.board,
                depth - 1,
                false,
                CONFIG.ALPHA,
                startTime,
                timeLimitMs,
                transTable
            ) + moveResult.score * 0.1;

            if (evalScore > depthBestScore) {
                depthBestScore = evalScore;
                depthBestAction = action;
            }

            // Time check between moves
            if (performance.now() - startTime > timeLimitMs * 0.95) break;
        }

        // Only update if we completed this depth
        if (performance.now() - startTime <= timeLimitMs * 0.95) {
            bestAction = depthBestAction;
            bestScore = depthBestScore;
            depthReached = depth;
        }

        // Time check between depths
        if (performance.now() - startTime > timeLimitMs * 0.8) break;
    }

    return {
        bestAction,
        score: bestScore,
        nodesSearched,
        depthReached,
        timeMs: performance.now() - startTime,
        empties: countEmpties(board),
    };
}

// =============================================================================
// Expectimax Bot Class (Compatible with AI Player interface)
// =============================================================================

export class ExpectimaxBot {
    private stats: BotStats = {
        gamesPlayed: 0,
        maxTileReached: 0,
        totalScore: 0,
        reached512: 0,
        reached1024: 0,
        reached2048: 0,
        reached4096: 0,
    };

    private lastSearchResult: SearchResult | null = null;
    private lastBoardHash: string = '';

    constructor() { }

    /**
     * Get the best action for the current board state.
     */
    getAction(board: number[], _legalActions: boolean[]): Action {
        const boardTyped = new Uint16Array(board);
        const result = findBestMove(boardTyped, CONFIG.TIME_LIMIT_MS);
        this.lastSearchResult = result;
        return result.bestAction;
    }

    /**
     * Async version for compatibility with AI player interface.
     */
    async getActionAsync(board: number[], _legalActions: boolean[]): Promise<Action> {
        return this.getAction(board, _legalActions);
    }

    /**
     * Check for stuck state (board didn't change after move).
     */
    checkBoardChanged(newBoard: number[]): boolean {
        const newHash = newBoard.join(',');
        const changed = newHash !== this.lastBoardHash;
        this.lastBoardHash = newHash;
        return changed;
    }

    /**
     * Record game end for stats.
     */
    recordGameEnd(score: number, maxTile: number): void {
        this.stats.gamesPlayed++;
        this.stats.totalScore += score;
        if (maxTile > this.stats.maxTileReached) {
            this.stats.maxTileReached = maxTile;
        }
        if (maxTile >= 512) this.stats.reached512++;
        if (maxTile >= 1024) this.stats.reached1024++;
        if (maxTile >= 2048) this.stats.reached2048++;
        if (maxTile >= 4096) this.stats.reached4096++;
    }

    /**
     * Get current stats.
     */
    getStats(): BotStats {
        return { ...this.stats };
    }

    /**
     * Get last search result for debugging.
     */
    getLastSearchResult(): SearchResult | null {
        return this.lastSearchResult;
    }

    /**
     * Reset stats.
     */
    resetStats(): void {
        this.stats = {
            gamesPlayed: 0,
            maxTileReached: 0,
            totalScore: 0,
            reached512: 0,
            reached1024: 0,
            reached2048: 0,
            reached4096: 0,
        };
    }

    // Interface compatibility methods
    isLoaded(): boolean { return true; }
    isLoading(): boolean { return false; }
    async load(): Promise<void> { }
    getName(): string { return 'Expectimax (Corner-Lock)'; }
    getType(): 'expectimax' { return 'expectimax'; }
}

// =============================================================================
// Action Names
// =============================================================================

export const ACTION_NAMES = ['↑ Up', '↓ Down', '← Left', '→ Right'];

export function getActionName(action: Action): string {
    return ACTION_NAMES[action];
}
