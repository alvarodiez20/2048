/**
 * Monte Carlo Tree Search (MCTS) Bot for 2048
 * 
 * Uses random rollouts to evaluate each possible move.
 * For each legal move, runs N simulations playing random moves to completion
 * and picks the initial move with the highest average final score.
 * 
 * Features:
 * - Configurable number of simulations per move
 * - Configurable max rollout depth
 * - Simple but effective — typically reaches 1024-2048
 * - No training required, works immediately
 */

// =============================================================================
// Types
// =============================================================================

export type Board = Uint16Array;
export type Action = 0 | 1 | 2 | 3; // Up, Down, Left, Right

export interface MCTSConfig {
    numSimulations: number;   // Rollouts per legal move (default: 200)
    maxRolloutDepth: number;  // Max random moves per rollout (default: 150)
}

interface MoveResult {
    board: Board;
    changed: boolean;
    score: number;
}

// =============================================================================
// Board Simulation (minimal — same logic as expectimax-bot)
// =============================================================================

function compressAndMerge(line: Uint16Array): number {
    let score = 0;
    let writeIdx = 0;
    for (let i = 0; i < 4; i++) {
        if (line[i] !== 0) line[writeIdx++] = line[i];
    }
    while (writeIdx < 4) line[writeIdx++] = 0;

    for (let i = 0; i < 3; i++) {
        if (line[i] !== 0 && line[i] === line[i + 1]) {
            line[i] *= 2;
            score += line[i];
            line[i + 1] = 0;
        }
    }

    writeIdx = 0;
    for (let i = 0; i < 4; i++) {
        if (line[i] !== 0) line[writeIdx++] = line[i];
    }
    while (writeIdx < 4) line[writeIdx++] = 0;

    return score;
}

function applyMove(board: Board, action: Action): MoveResult {
    const newBoard = new Uint16Array(board);
    let totalScore = 0;
    let changed = false;
    const line = new Uint16Array(4);

    if (action === 0) { // Up
        for (let col = 0; col < 4; col++) {
            for (let row = 0; row < 4; row++) line[row] = newBoard[row * 4 + col];
            const old = Uint16Array.from(line);
            totalScore += compressAndMerge(line);
            for (let row = 0; row < 4; row++) newBoard[row * 4 + col] = line[row];
            if (!changed) for (let r = 0; r < 4; r++) if (old[r] !== line[r]) { changed = true; break; }
        }
    } else if (action === 1) { // Down
        for (let col = 0; col < 4; col++) {
            for (let row = 0; row < 4; row++) line[row] = newBoard[(3 - row) * 4 + col];
            const old = Uint16Array.from(line);
            totalScore += compressAndMerge(line);
            for (let row = 0; row < 4; row++) newBoard[(3 - row) * 4 + col] = line[row];
            if (!changed) for (let r = 0; r < 4; r++) if (old[r] !== line[r]) { changed = true; break; }
        }
    } else if (action === 2) { // Left
        for (let row = 0; row < 4; row++) {
            for (let col = 0; col < 4; col++) line[col] = newBoard[row * 4 + col];
            const old = Uint16Array.from(line);
            totalScore += compressAndMerge(line);
            for (let col = 0; col < 4; col++) newBoard[row * 4 + col] = line[col];
            if (!changed) for (let c = 0; c < 4; c++) if (old[c] !== line[c]) { changed = true; break; }
        }
    } else { // Right
        for (let row = 0; row < 4; row++) {
            for (let col = 0; col < 4; col++) line[col] = newBoard[row * 4 + (3 - col)];
            const old = Uint16Array.from(line);
            totalScore += compressAndMerge(line);
            for (let col = 0; col < 4; col++) newBoard[row * 4 + (3 - col)] = line[col];
            if (!changed) for (let c = 0; c < 4; c++) if (old[c] !== line[c]) { changed = true; break; }
        }
    }

    return { board: newBoard, changed, score: totalScore };
}

function getLegalMoves(board: Board): Action[] {
    const legal: Action[] = [];
    for (const action of [0, 1, 2, 3] as Action[]) {
        if (applyMove(board, action).changed) legal.push(action);
    }
    return legal;
}

function spawnRandomTile(board: Board): void {
    const empties: number[] = [];
    for (let i = 0; i < 16; i++) {
        if (board[i] === 0) empties.push(i);
    }
    if (empties.length === 0) return;
    const idx = empties[Math.floor(Math.random() * empties.length)];
    board[idx] = Math.random() < 0.9 ? 2 : 4;
}

function getMaxTile(board: Board): number {
    let max = 0;
    for (let i = 0; i < 16; i++) {
        if (board[i] > max) max = board[i];
    }
    return max;
}

// =============================================================================
// MCTS Core
// =============================================================================

/**
 * Run a single random rollout from the given board state.
 * Returns the total score accumulated during the rollout.
 */
function randomRollout(board: Board, maxDepth: number): number {
    const simBoard = new Uint16Array(board);
    let totalScore = 0;

    for (let depth = 0; depth < maxDepth; depth++) {
        const legal = getLegalMoves(simBoard);
        if (legal.length === 0) break;

        const action = legal[Math.floor(Math.random() * legal.length)];
        const result = applyMove(simBoard, action);
        totalScore += result.score;

        // Copy result back
        simBoard.set(result.board);
        spawnRandomTile(simBoard);
    }

    // Add a heuristic bonus for the max tile reached
    const maxTile = getMaxTile(simBoard);
    totalScore += maxTile;

    return totalScore;
}

/**
 * Find the best move using Monte Carlo simulations.
 */
function findBestMoveMCTS(board: Board, config: MCTSConfig): Action {
    const legalMoves = getLegalMoves(board);
    if (legalMoves.length === 0) return 0;
    if (legalMoves.length === 1) return legalMoves[0];

    let bestAction = legalMoves[0];
    let bestAvgScore = -Infinity;

    for (const action of legalMoves) {
        const moveResult = applyMove(board, action);
        let totalScore = 0;

        for (let sim = 0; sim < config.numSimulations; sim++) {
            // Start rollout from the post-move board (with a random spawn)
            const rolloutBoard = new Uint16Array(moveResult.board);
            spawnRandomTile(rolloutBoard);

            totalScore += moveResult.score + randomRollout(rolloutBoard, config.maxRolloutDepth);
        }

        const avgScore = totalScore / config.numSimulations;
        if (avgScore > bestAvgScore) {
            bestAvgScore = avgScore;
            bestAction = action;
        }
    }

    return bestAction;
}

// =============================================================================
// MCTS Bot Class (Compatible with AI Player interface)
// =============================================================================

export class MCTSBot {
    private config: MCTSConfig;

    constructor(numSimulations: number = 200, maxRolloutDepth: number = 150) {
        this.config = { numSimulations, maxRolloutDepth };
    }

    /**
     * Get the best action for the current board state.
     */
    getAction(board: number[], _legalActions: boolean[]): Action {
        const boardTyped = new Uint16Array(board);
        return findBestMoveMCTS(boardTyped, this.config);
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
    getName(): string { return 'MCTS (Monte Carlo)'; }
    getType(): 'mcts' { return 'mcts'; }
}

// =============================================================================
// Action Names
// =============================================================================

export const ACTION_NAMES = ['↑ Up', '↓ Down', '← Left', '→ Right'];

export function getActionName(action: Action): string {
    return ACTION_NAMES[action];
}
