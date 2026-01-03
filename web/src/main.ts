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
import { ExpectimaxBot } from './expectimax-bot';
import { RustSolverBot } from './rust-solver-bot';

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
let aiPlayer: AIPlayerType | ExpectimaxBot | RustSolverBot | null = null;
let aiRunning = false;
let aiInterval: number | null = null;
let expectimaxBot: ExpectimaxBot | null = null;
let rustBot: RustSolverBot | null = null;

// Canvas dimensions
let CANVAS_SIZE = 400;
const GRID_SIZE = 4;
const PADDING = 12;
const CELL_GAP = 10;
let CELL_SIZE = (CANVAS_SIZE - 2 * PADDING - (GRID_SIZE - 1) * CELL_GAP) / GRID_SIZE;

// =============================================================================
// Initialization
// =============================================================================

async function main() {
    // Initialize WASM module
    await init();

    // Get DOM elements
    canvas = document.getElementById('game-canvas') as HTMLCanvasElement;
    ctx = canvas.getContext('2d')!;

    // Set initial canvas size responsively
    updateCanvasSize();
    window.addEventListener('resize', () => {
        updateCanvasSize();
        render();
    });

    // Initialize game
    const seedInput = document.getElementById('seed-input') as HTMLInputElement;
    startNewGame(parseInt(seedInput.value) || 42);

    // Setup event listeners
    setupEventListeners();

    // Initialize AI
    await initializeAI();
}

function updateCanvasSize() {
    if (!canvas) return;

    // Calculate size based on viewport
    let size = 400;

    if (window.innerWidth <= 768) {
        // Mobile: 90vw but max 400px
        size = Math.min(Math.floor(window.innerWidth * 0.9), 400);
    }

    // Update global constants
    CANVAS_SIZE = size;
    CELL_SIZE = (CANVAS_SIZE - 2 * PADDING - (GRID_SIZE - 1) * CELL_GAP) / GRID_SIZE;

    // Set canvas size (both internal and displayed size match to avoid distortion)
    canvas.width = size;
    canvas.height = size;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
}

async function initializeAI() {
    const statusEl = document.getElementById('ai-status');
    const statusText = document.getElementById('ai-status-text');
    const watchBtn = document.getElementById('watch-ai') as HTMLButtonElement;
    const hintBtn = document.getElementById('ai-hint') as HTMLButtonElement;
    const modelSelect = document.getElementById('model-select') as HTMLSelectElement;

    // Load model manifest
    try {
        const { loadModelManifest } = await import('./ai-player');
        const models = await loadModelManifest('models/');

        // Add Rust Solver (best performance)
        const rustOption = document.createElement('option');
        rustOption.value = 'rust-solver';
        rustOption.textContent = 'Rust Solver (Bitboard)';
        rustOption.dataset.description = 'High-performance expectimax with bitboards [FASTEST]';
        modelSelect.appendChild(rustOption);

        // Add Expectimax bot
        const expectimaxOption = document.createElement('option');
        expectimaxOption.value = 'expectimax';
        expectimaxOption.textContent = 'Expectimax (Corner-Lock)';
        expectimaxOption.dataset.description = 'Corner-locked strategy with risk-averse search';
        modelSelect.appendChild(expectimaxOption);

        // Add trained models from manifest (filter for best ones)
        const desiredModels = ['dqn_shaped', 'dqn_cnn'];
        for (const model of models.filter(m => desiredModels.includes(m.id))) {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.name;
            option.dataset.type = model.type;
            option.dataset.file = model.file;
            option.dataset.description = model.description;
            modelSelect.appendChild(option);
        }

        // Model change handler
        modelSelect.addEventListener('change', async () => {
            await loadSelectedModel();
            updateModelDescription();
        });

        // Initial description update
        updateModelDescription();

        // Load first available model or random
        if (models.length > 0) {
            modelSelect.value = models[0].id;
        }
        await loadSelectedModel();

    } catch (error) {
        console.error('Failed to initialize AI:', error);
        aiPlayer = new RandomAIPlayer();
        if (statusEl) statusEl.className = 'ai-status ready';
        if (statusText) statusText.textContent = 'AI ready (random mode)';
        if (watchBtn) watchBtn.disabled = false;
        if (hintBtn) hintBtn.disabled = false;
    }
}

async function loadSelectedModel() {
    const statusEl = document.getElementById('ai-status');
    const statusText = document.getElementById('ai-status-text');
    const watchBtn = document.getElementById('watch-ai') as HTMLButtonElement;
    const hintBtn = document.getElementById('ai-hint') as HTMLButtonElement;
    const modelSelect = document.getElementById('model-select') as HTMLSelectElement;

    // Stop AI if running
    stopAI();

    const selectedOption = modelSelect.options[modelSelect.selectedIndex];
    const modelId = selectedOption.value;

    if (statusEl) statusEl.className = 'ai-status';
    if (statusText) statusText.textContent = 'Loading model...';
    if (watchBtn) watchBtn.disabled = true;
    if (hintBtn) hintBtn.disabled = true;

    try {
        if (modelId === 'random') {
            aiPlayer = new RandomAIPlayer();
            expectimaxBot = null;
            rustBot = null;
            if (statusEl) statusEl.className = 'ai-status ready';
            if (statusText) statusText.textContent = 'Random baseline';
        } else if (modelId === 'rust-solver') {
            rustBot = new RustSolverBot(100); // 100ms time limit
            aiPlayer = rustBot;
            expectimaxBot = null;
            if (statusEl) statusEl.className = 'ai-status ready';
            if (statusText) statusText.textContent = 'Rust Solver ready';
        } else if (modelId === 'expectimax') {
            expectimaxBot = new ExpectimaxBot();
            aiPlayer = expectimaxBot;
            rustBot = null;
            if (statusEl) statusEl.className = 'ai-status ready';
            if (statusText) statusText.textContent = 'Expectimax ready';
        } else {
            const modelType = (selectedOption.dataset.type || 'mlp') as 'mlp' | 'cnn';
            const modelFile = selectedOption.dataset.file || `${modelId}.onnx`;

            const { AIPlayer: AIPlayerClass } = await import('./ai-player');
            aiPlayer = new AIPlayerClass(`models/${modelFile}`, modelType);
            await aiPlayer.load();
            expectimaxBot = null;
            rustBot = null;

            if (statusEl) statusEl.className = 'ai-status ready';
            if (statusText) statusText.textContent = `${selectedOption.textContent} ready`;
        }

        if (watchBtn) watchBtn.disabled = false;
        if (hintBtn) hintBtn.disabled = false;

    } catch (error) {
        console.error('Failed to load model:', error);
        aiPlayer = new RandomAIPlayer();
        if (statusEl) statusEl.className = 'ai-status error';
        if (statusText) statusText.textContent = 'Model failed, using random';
        if (watchBtn) watchBtn.disabled = false;
        if (hintBtn) hintBtn.disabled = false;
    }
}

function updateModelDescription() {
    const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
    const descriptionEl = document.getElementById('model-description');

    if (!modelSelect || !descriptionEl) return;

    const selectedOption = modelSelect.options[modelSelect.selectedIndex];
    const description = selectedOption.dataset.description;

    const descriptions: Record<string, string> = {
        'random': '🎲 Simple baseline that picks moves randomly',
        'rust-solver': '⚡ High-performance expectimax using bitboards [FASTEST]',
        'expectimax': '🧠 Corner-locked strategy with risk-averse search',
        'dqn_shaped': '🎯 Deep Q-Network with reward shaping (trained on RL)',
        'dqn_cnn': '🖼️ Convolutional Neural Network trained via Deep RL'
    };

    const modelId = selectedOption.value;
    descriptionEl.textContent = description || descriptions[modelId] || 'AI model';
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

    // Touch/swipe support for mobile
    setupTouchControls();

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

// =============================================================================
// Touch Controls for Mobile
// =============================================================================

let touchStartX = 0;
let touchStartY = 0;
let touchEndX = 0;
let touchEndY = 0;

const MIN_SWIPE_DISTANCE = 50; // Minimum distance for a swipe to register

function setupTouchControls() {
    const canvas = document.getElementById('game-canvas');
    if (!canvas) return;

    canvas.addEventListener('touchstart', handleTouchStart, { passive: false });
    canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
    canvas.addEventListener('touchend', handleTouchEnd, { passive: false });
}

function handleTouchStart(e: TouchEvent) {
    e.preventDefault(); // Prevent scrolling
    const touch = e.touches[0];
    touchStartX = touch.clientX;
    touchStartY = touch.clientY;
}

function handleTouchMove(e: TouchEvent) {
    e.preventDefault(); // Prevent scrolling
}

function handleTouchEnd(e: TouchEvent) {
    e.preventDefault();

    // Stop AI when user swipes
    if (aiRunning) {
        stopAI();
    }

    if (!game || game.isDone()) return;

    const touch = e.changedTouches[0];
    touchEndX = touch.clientX;
    touchEndY = touch.clientY;

    const deltaX = touchEndX - touchStartX;
    const deltaY = touchEndY - touchStartY;

    let action: number | null = null;

    // Determine if horizontal or vertical swipe
    if (Math.abs(deltaX) > Math.abs(deltaY)) {
        // Horizontal swipe
        if (Math.abs(deltaX) > MIN_SWIPE_DISTANCE) {
            if (deltaX > 0) {
                action = Action.Right;
                showSwipeIndicator('→');
            } else {
                action = Action.Left;
                showSwipeIndicator('←');
            }
        }
    } else {
        // Vertical swipe
        if (Math.abs(deltaY) > MIN_SWIPE_DISTANCE) {
            if (deltaY > 0) {
                action = Action.Down;
                showSwipeIndicator('↓');
            } else {
                action = Action.Up;
                showSwipeIndicator('↑');
            }
        }
    }

    if (action !== null) {
        performAction(action);
    }
}

// Visual feedback for swipe direction
function showSwipeIndicator(arrow: string) {
    const canvas = document.getElementById('game-canvas');
    if (!canvas) return;

    const indicator = document.createElement('div');
    indicator.textContent = arrow;
    indicator.style.cssText = `
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 4rem;
        color: rgba(233, 69, 96, 0.8);
        pointer-events: none;
        z-index: 100;
        animation: swipeFeedback 0.3s ease-out;
    `;

    canvas.parentElement?.appendChild(indicator);
    setTimeout(() => indicator.remove(), 300);
}

// Add swipe animation
const style = document.createElement('style');
style.textContent = `
    @keyframes swipeFeedback {
        0% {
            opacity: 0;
            transform: translate(-50%, -50%) scale(0.5);
        }
        50% {
            opacity: 1;
            transform: translate(-50%, -50%) scale(1.2);
        }
        100% {
            opacity: 0;
            transform: translate(-50%, -50%) scale(1);
        }
    }
`;
document.head.appendChild(style);

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
        if (!game || !aiPlayer) {
            stopAI();
            return;
        }

        if (game.isDone()) {
            // Record stats for expectimax bot
            if (expectimaxBot) {
                const board = Array.from(game.getBoard());
                const maxTile = Math.max(...board);
                expectimaxBot.recordGameEnd(game.getScore(), maxTile);
            }
            stopAI();
            return;
        }

        try {
            const board = Array.from(game.getBoard());
            const legalActions = Array.from(game.getLegalActions()).map(v => v === 1);

            let action: AIAction;
            if (aiPlayer instanceof RandomAIPlayer) {
                action = aiPlayer.getAction(legalActions);
            } else if (rustBot && aiPlayer === rustBot) {
                action = rustBot.getAction(board, legalActions) as AIAction;
            } else if (expectimaxBot && aiPlayer === expectimaxBot) {
                action = expectimaxBot.getAction(board, legalActions) as AIAction;
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

