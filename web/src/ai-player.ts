/**
 * AI Player for 2048 using ONNX.js
 * 
 * This module provides AI player functionality using a trained DQN model
 * exported to ONNX format.
 */

import * as ort from 'onnxruntime-web';

export type AIAction = 0 | 1 | 2 | 3; // Up, Down, Left, Right
export type QValues = [number, number, number, number];

export class AIPlayer {
    private session: ort.InferenceSession | null = null;
    private modelPath: string;
    private loading: boolean = false;
    private loaded: boolean = false;

    constructor(modelPath: string = '/models/ai_model.onnx') {
        this.modelPath = modelPath;
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

        // Normalize board state (log2 / 17)
        const normalizedBoard = board.map(v =>
            v === 0 ? 0 : Math.log2(v) / 17
        );

        // Create input tensor
        const inputTensor = new ort.Tensor('float32', normalizedBoard, [1, 16]);

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
     * 
     * @param board - Current board state (16 values)
     * @returns Q-values for [Up, Down, Left, Right]
     */
    async getQValues(board: number[]): Promise<QValues> {
        if (!this.session) {
            throw new Error('AI model not loaded. Call load() first.');
        }

        // Normalize board state
        const normalizedBoard = board.map(v =>
            v === 0 ? 0 : Math.log2(v) / 17
        );

        // Create input tensor
        const inputTensor = new ort.Tensor('float32', normalizedBoard, [1, 16]);

        // Run inference
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
}

export type AIPlayerType = AIPlayer | RandomAIPlayer;
