/**
 * AI Player for 2048 using ONNX.js
 * 
 * This module provides AI player functionality using trained DQN models
 * exported to ONNX format. Supports both MLP and CNN architectures.
 */

import * as ort from 'onnxruntime-web';

export type AIAction = 0 | 1 | 2 | 3; // Up, Down, Left, Right
export type QValues = [number, number, number, number];
export type ModelType = 'mlp' | 'cnn';

export interface ModelInfo {
    id: string;
    name: string;
    description: string;
    file: string;
    type: ModelType;
}

export class AIPlayer {
    private session: ort.InferenceSession | null = null;
    private modelPath: string;
    private modelType: ModelType;
    private loading: boolean = false;
    private loaded: boolean = false;
    private modelName: string = '';

    constructor(modelPath: string = 'models/ai_model.onnx', modelType: ModelType = 'mlp') {
        this.modelPath = modelPath;
        this.modelType = modelType;
    }

    /**
     * Load the ONNX model.
     * Call this before using getAction().
     */
    async load(): Promise<void> {
        if (this.loaded || this.loading) return;

        this.loading = true;
        try {
            console.log(`Loading AI model from ${this.modelPath}...`);
            this.session = await ort.InferenceSession.create(this.modelPath);
            this.loaded = true;
            console.log('AI model loaded successfully!');
        } catch (error) {
            console.error('Failed to load AI model:', error);
            throw error;
        } finally {
            this.loading = false;
        }
    }

    /**
     * Create an AIPlayer from a ModelInfo object.
     */
    static fromModelInfo(info: ModelInfo, basePath: string = 'models/'): AIPlayer {
        const player = new AIPlayer(basePath + info.file, info.type);
        player.modelName = info.name;
        return player;
    }

    /**
     * Check if the model is loaded.
     */
    isLoaded(): boolean {
        return this.loaded;
    }

    /**
     * Check if the model is currently loading.
     */
    isLoading(): boolean {
        return this.loading;
    }

    /**
     * Get the model name.
     */
    getName(): string {
        return this.modelName || 'AI Model';
    }

    /**
     * Get the model type.
     */
    getType(): ModelType {
        return this.modelType;
    }

    /**
     * Encode board for CNN model (one-hot encoding).
     */
    private encodeBoardCNN(board: number[]): Float32Array {
        // 16 channels x 4 x 4 = 256 values
        const encoded = new Float32Array(16 * 4 * 4);

        for (let i = 0; i < 16; i++) {
            const row = Math.floor(i / 4);
            const col = i % 4;
            const val = board[i];

            let channel = 0;
            if (val > 0) {
                channel = Math.min(15, Math.floor(Math.log2(val)));
            }

            // Index: channel * 16 + row * 4 + col
            encoded[channel * 16 + row * 4 + col] = 1.0;
        }

        return encoded;
    }

    /**
     * Encode board for MLP model (log2 normalized).
     */
    private encodeBoardMLP(board: number[]): Float32Array {
        return new Float32Array(board.map(v =>
            v === 0 ? 0 : Math.log2(v) / 17
        ));
    }

    /**
     * Get the best action for the current board state.
     * 
     * @param board - Current board state (16 values)
     * @param legalActions - Boolean array of legal actions [Up, Down, Left, Right]
     * @returns Best action (0-3)
     */
    async getAction(board: number[], legalActions: boolean[]): Promise<AIAction> {
        if (!this.session) {
            throw new Error('AI model not loaded. Call load() first.');
        }

        // Encode board based on model type
        let inputTensor: ort.Tensor;
        if (this.modelType === 'cnn') {
            const encoded = this.encodeBoardCNN(board);
            inputTensor = new ort.Tensor('float32', encoded, [1, 16, 4, 4]);
        } else {
            const encoded = this.encodeBoardMLP(board);
            inputTensor = new ort.Tensor('float32', encoded, [1, 16]);
        }

        // Run inference
        const outputs = await this.session.run({ input: inputTensor });
        const qValues = outputs['q_values'].data as Float32Array;

        // Find best legal action
        let bestAction: AIAction = 0;
        let bestValue = -Infinity;

        for (let i = 0; i < 4; i++) {
            if (legalActions[i] && qValues[i] > bestValue) {
                bestValue = qValues[i];
                bestAction = i as AIAction;
            }
        }

        return bestAction;
    }

    /**
     * Get Q-values for all actions.
     */
    async getQValues(board: number[]): Promise<QValues> {
        if (!this.session) {
            throw new Error('AI model not loaded. Call load() first.');
        }

        let inputTensor: ort.Tensor;
        if (this.modelType === 'cnn') {
            const encoded = this.encodeBoardCNN(board);
            inputTensor = new ort.Tensor('float32', encoded, [1, 16, 4, 4]);
        } else {
            const encoded = this.encodeBoardMLP(board);
            inputTensor = new ort.Tensor('float32', encoded, [1, 16]);
        }

        const outputs = await this.session.run({ input: inputTensor });
        const qValues = outputs['q_values'].data as Float32Array;

        return [qValues[0], qValues[1], qValues[2], qValues[3]];
    }

    /**
     * Get action names for display.
     */
    static getActionName(action: AIAction): string {
        const names = ['Up', 'Down', 'Left', 'Right'];
        return names[action];
    }
}

/**
 * Simple random AI player (fallback when model not available).
 */
export class RandomAIPlayer {
    getAction(legalActions: boolean[]): AIAction {
        const validActions: AIAction[] = [];
        for (let i = 0; i < 4; i++) {
            if (legalActions[i]) {
                validActions.push(i as AIAction);
            }
        }

        if (validActions.length === 0) return 0;
        return validActions[Math.floor(Math.random() * validActions.length)];
    }

    isLoaded(): boolean {
        return true;
    }

    isLoading(): boolean {
        return false;
    }

    async load(): Promise<void> {
        // No-op for random player
    }

    getName(): string {
        return 'Random';
    }

    getType(): ModelType {
        return 'mlp';
    }
}

export type AIPlayerType = AIPlayer | RandomAIPlayer;

/**
 * Load model manifest from server.
 */
export async function loadModelManifest(basePath: string = 'models/'): Promise<ModelInfo[]> {
    try {
        const response = await fetch(basePath + 'manifest.json');
        if (!response.ok) {
            console.warn('Model manifest not found, using default model');
            return [];
        }
        const manifest = await response.json();
        return manifest.models || [];
    } catch (error) {
        console.warn('Failed to load model manifest:', error);
        return [];
    }
}
