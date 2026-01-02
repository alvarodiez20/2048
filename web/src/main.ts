/**
 * 2048 Game - Main TypeScript Entry Point
 *
 * This module handles:
 * - WASM module initialization
 * - Canvas rendering of the game board
 * - Keyboard input handling
 * - UI updates (score, game over overlay)
 * - AI player integration
 */

import init, { WasmGame } from './wasm-pkg/game_2048_wasm.js';
import { AIPlayer, RandomAIPlayer, AIPlayerType, AIAction } from './ai-player';

// =============================================================================
// Types
// =============================================================================

interface StepResult {
    board: number[];
    score: number;
    reward: number;
    changed: boolean;
    done: boolean;
}

// Action mapping: matches core engine
const Action = {
    Up: 0,
    Down: 1,
    Left: 2,
    Right: 3,
} as const;

const ActionNames = ['↑ Up', '↓ Down', '← Left', '→ Right'];

// Tile colors matching the classic 2048 palette
const TILE_COLORS: Record<number, { bg: string; text: string }> = {
    0: { bg: 'rgba(255, 255, 255, 0.05)', text: '#776e65' },
    2: { bg: '#eee4da', text: '#776e65' },
    4: { bg: '#ede0c8', text: '#776e65' },
    8: { bg: '#f2b179', text: '#f9f6f2' },
    16: { bg: '#f59563', text: '#f9f6f2' },
    32: { bg: '#f67c5f', text: '#f9f6f2' },
    64: { bg: '#f65e3b', text: '#f9f6f2' },
    128: { bg: '#edcf72', text: '#f9f6f2' },
    256: { bg: '#edcc61', text: '#f9f6f2' },
    512: { bg: '#edc850', text: '#f9f6f2' },
    1024: { bg: '#edc53f', text: '#f9f6f2' },
    2048: { bg: '#edc22e', text: '#f9f6f2' },
};

// Default color for tiles > 2048
const SUPER_TILE_COLOR = { bg: '#3c3a32', text: '#f9f6f2' };

// =============================================================================
// Game State
// =============================================================================

let game: WasmGame | null = null;
let canvas: HTMLCanvasElement;
let ctx: CanvasRenderingContext2D;

// AI State
let aiPlayer: AIPlayerType | null = null;
let aiRunning = false;
let aiInterval: number | null = null;

// Canvas dimensions
const CANVAS_SIZE = 400;
const GRID_SIZE = 4;
const PADDING = 12;
const CELL_GAP = 10;
const CELL_SIZE = (CANVAS_SIZE - 2 * PADDING - (GRID_SIZE - 1) * CELL_GAP) / GRID_SIZE;

// =============================================================================
// Initialization
// =============================================================================

async function main() {
    // Initialize WASM module
    await init();

    // Get DOM elements
    canvas = document.getElementById('game-canvas') as HTMLCanvasElement;
    ctx = canvas.getContext('2d')!;

    // Set canvas size
    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;

    // Initialize game
    const seedInput = document.getElementById('seed-input') as HTMLInputElement;
    startNewGame(parseInt(seedInput.value) || 42);

    // Setup event listeners
    setupEventListeners();

    // Initialize AI
    await initializeAI();
}

async function initializeAI() {
    const statusEl = document.getElementById('ai-status');
    const statusText = document.getElementById('ai-status-text');
    const watchBtn = document.getElementById('watch-ai') as HTMLButtonElement;
    const hintBtn = document.getElementById('ai-hint') as HTMLButtonElement;

    try {
        // Try to load the trained model
        aiPlayer = new AIPlayer('/models/ai_model.onnx');
        await aiPlayer.load();

        if (statusEl) statusEl.className = 'ai-status ready';
        if (statusText) statusText.textContent = 'AI ready';
        if (watchBtn) watchBtn.disabled = false;
        if (hintBtn) hintBtn.disabled = false;
    } catch (error) {
        console.warn('Trained AI model not found, using random player:', error);

        // Fall back to random player
        aiPlayer = new RandomAIPlayer();

        if (statusEl) statusEl.className = 'ai-status ready';
        if (statusText) statusText.textContent = 'AI ready (random mode)';
        if (watchBtn) watchBtn.disabled = false;
        if (hintBtn) hintBtn.disabled = false;
    }
}

function startNewGame(seed: number) {
    // Stop AI if running
    stopAI();

    game = new WasmGame(BigInt(seed));
    hideGameOver();
    hideHint();
    updateScore(0);
    render();
}

// =============================================================================
// Event Handling
// =============================================================================

function setupEventListeners() {
    // Keyboard input
    document.addEventListener('keydown', handleKeyDown);

    // New game button
    document.getElementById('new-game')?.addEventListener('click', () => {
        const seedInput = document.getElementById('seed-input') as HTMLInputElement;
        startNewGame(parseInt(seedInput.value) || 42);
    });

    // Restart button (in game over overlay)
    document.getElementById('restart')?.addEventListener('click', () => {
        const seedInput = document.getElementById('seed-input') as HTMLInputElement;
        startNewGame(parseInt(seedInput.value) || 42);
    });

    // Prevent arrow keys from scrolling the page
    window.addEventListener('keydown', (e) => {
        if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
            e.preventDefault();
        }
    });

    // AI controls
    document.getElementById('watch-ai')?.addEventListener('click', startAI);
    document.getElementById('stop-ai')?.addEventListener('click', stopAI);
    document.getElementById('ai-hint')?.addEventListener('click', showHint);

    // Speed slider
    const speedSlider = document.getElementById('ai-speed-slider') as HTMLInputElement;
    const speedValue = document.getElementById('speed-value');
    speedSlider?.addEventListener('input', () => {
        if (speedValue) speedValue.textContent = speedSlider.value;

        // Update AI interval if running
        if (aiRunning) {
            stopAI();
            startAI();
        }
    });
}

function handleKeyDown(e: KeyboardEvent) {
    // Stop AI when user presses a key
    if (aiRunning) {
        stopAI();
    }

    if (!game || game.isDone()) return;

    let action: number | null = null;

    switch (e.key) {
        case 'ArrowUp':
        case 'w':
        case 'W':
            action = Action.Up;
            break;
        case 'ArrowDown':
        case 's':
        case 'S':
            action = Action.Down;
            break;
        case 'ArrowLeft':
        case 'a':
        case 'A':
            action = Action.Left;
            break;
        case 'ArrowRight':
        case 'd':
        case 'D':
            action = Action.Right;
            break;
    }

    if (action !== null) {
        performAction(action);
    }
}

function performAction(action: number) {
    if (!game) return;

    const result = game.step(action) as StepResult;

    if (result.changed) {
        updateScore(result.score);
        render();

        if (result.done) {
            stopAI();
            showGameOver(result.score, getMaxTile(result.board));
        }
    }

    return result;
}

// =============================================================================
// AI Controls
// =============================================================================

function startAI() {
    if (!game || !aiPlayer || game.isDone()) return;

    aiRunning = true;

    // Update UI
    const watchBtn = document.getElementById('watch-ai');
    const stopBtn = document.getElementById('stop-ai');
    const statusEl = document.getElementById('ai-status');
    const statusText = document.getElementById('ai-status-text');

    if (watchBtn) watchBtn.classList.add('hidden');
    if (stopBtn) stopBtn.classList.remove('hidden');
    if (statusEl) statusEl.className = 'ai-status running';
    if (statusText) statusText.textContent = 'AI playing...';

    hideHint();

    // Get speed from slider
    const speedSlider = document.getElementById('ai-speed-slider') as HTMLInputElement;
    const movesPerSecond = parseInt(speedSlider?.value || '5');
    const intervalMs = 1000 / movesPerSecond;

    // Start AI loop
    aiInterval = window.setInterval(async () => {
        if (!game || !aiPlayer || game.isDone()) {
            stopAI();
            return;
        }

        try {
            const board = Array.from(game.getBoard());
            const legalActions = Array.from(game.getLegalActions()).map(v => v === 1);

            let action: AIAction;
            if (aiPlayer instanceof RandomAIPlayer) {
                action = aiPlayer.getAction(legalActions);
            } else {
                action = await (aiPlayer as AIPlayer).getAction(board, legalActions);
            }

            performAction(action);
        } catch (error) {
            console.error('AI error:', error);
            stopAI();
        }
    }, intervalMs);
}

function stopAI() {
    if (aiInterval !== null) {
        clearInterval(aiInterval);
        aiInterval = null;
    }

    aiRunning = false;

    // Update UI
    const watchBtn = document.getElementById('watch-ai');
    const stopBtn = document.getElementById('stop-ai');
    const statusEl = document.getElementById('ai-status');
    const statusText = document.getElementById('ai-status-text');

    if (watchBtn) watchBtn.classList.remove('hidden');
    if (stopBtn) stopBtn.classList.add('hidden');
    if (statusEl) statusEl.className = 'ai-status ready';
    if (statusText) statusText.textContent = aiPlayer instanceof RandomAIPlayer ? 'AI ready (random mode)' : 'AI ready';
}

async function showHint() {
    if (!game || !aiPlayer || game.isDone()) return;

    try {
        const board = Array.from(game.getBoard());
        const legalActions = Array.from(game.getLegalActions()).map(v => v === 1);

        let action: AIAction;
        if (aiPlayer instanceof RandomAIPlayer) {
            action = aiPlayer.getAction(legalActions);
        } else {
            action = await (aiPlayer as AIPlayer).getAction(board, legalActions);
        }

        // Show hint
        const hintDisplay = document.getElementById('ai-hint-display');
        const hintAction = document.getElementById('hint-action');

        if (hintDisplay) hintDisplay.classList.remove('hidden');
        if (hintAction) hintAction.textContent = ActionNames[action];
    } catch (error) {
        console.error('Hint error:', error);
    }
}

function hideHint() {
    const hintDisplay = document.getElementById('ai-hint-display');
    if (hintDisplay) hintDisplay.classList.add('hidden');
}

// =============================================================================
// Rendering
// =============================================================================

function render() {
    if (!game) return;

    const board = game.getBoard();

    // Clear canvas
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // Draw grid background
    ctx.fillStyle = 'rgba(255, 255, 255, 0.03)';
    roundRect(ctx, PADDING / 2, PADDING / 2, CANVAS_SIZE - PADDING, CANVAS_SIZE - PADDING, 12);
    ctx.fill();

    // Draw tiles
    for (let row = 0; row < GRID_SIZE; row++) {
        for (let col = 0; col < GRID_SIZE; col++) {
            const index = row * GRID_SIZE + col;
            const value = board[index];
            drawTile(col, row, value);
        }
    }
}

function drawTile(col: number, row: number, value: number) {
    const x = PADDING + col * (CELL_SIZE + CELL_GAP);
    const y = PADDING + row * (CELL_SIZE + CELL_GAP);

    // Get tile color
    const colors = TILE_COLORS[value] || SUPER_TILE_COLOR;

    // Draw tile background
    ctx.fillStyle = colors.bg;
    roundRect(ctx, x, y, CELL_SIZE, CELL_SIZE, 8);
    ctx.fill();

    // Draw tile value
    if (value > 0) {
        ctx.fillStyle = colors.text;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // Adjust font size based on number of digits
        const fontSize = value >= 1000 ? 28 : value >= 100 ? 34 : 42;
        ctx.font = `bold ${fontSize}px 'Segoe UI', system-ui, sans-serif`;

        ctx.fillText(value.toString(), x + CELL_SIZE / 2, y + CELL_SIZE / 2);
    }
}

function roundRect(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    width: number,
    height: number,
    radius: number
) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
}

// =============================================================================
// UI Updates
// =============================================================================

function updateScore(score: number) {
    const scoreEl = document.getElementById('score');
    if (scoreEl) {
        scoreEl.textContent = score.toString();
        scoreEl.classList.remove('score-pop');
        // Trigger reflow to restart animation
        void scoreEl.offsetWidth;
        scoreEl.classList.add('score-pop');
    }
}

function showGameOver(score: number, maxTile: number) {
    const overlay = document.getElementById('game-over');
    const finalScore = document.getElementById('final-score');
    const maxTileEl = document.getElementById('max-tile');

    if (overlay) overlay.classList.remove('hidden');
    if (finalScore) finalScore.textContent = score.toString();
    if (maxTileEl) maxTileEl.textContent = maxTile.toString();
}

function hideGameOver() {
    const overlay = document.getElementById('game-over');
    if (overlay) overlay.classList.add('hidden');
}

function getMaxTile(board: number[]): number {
    return Math.max(...board);
}

// =============================================================================
// Start the game
// =============================================================================

main().catch(console.error);
