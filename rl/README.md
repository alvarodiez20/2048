# 2048 RL Training

This directory contains the reinforcement learning training pipeline for the 2048 AI.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install Rust Python bindings for faster training
cd ../python
pip install maturin
maturin develop --release
cd ../rl
```

## Quick Start

### Training

```bash
# Basic training (10k episodes)
python train.py --episodes 10000 --seed 42

# Extended training with custom parameters
python train.py \
    --episodes 50000 \
    --lr 1e-4 \
    --batch-size 256 \
    --epsilon-decay 100000 \
    --log-dir runs/experiment1

# Monitor training with TensorBoard
tensorboard --logdir runs
```

### Evaluation

```bash
# Evaluate a trained model
python evaluate.py --model checkpoints/best_model.pt --episodes 100

# Compare with random baseline
python evaluate.py --model checkpoints/best_model.pt --episodes 100 --compare-random
```

### Export for Web

```bash
# Export to ONNX
python export_onnx.py --model checkpoints/best_model.pt --output ../web/public/models/ai_model.onnx

# Export with quantization (smaller file)
python export_onnx.py --model checkpoints/best_model.pt --output ai_model.onnx --quantize
```

## Files

- `game_env.py` - Pure Python 2048 environment (mirrors Rust logic)
- `dqn_agent.py` - DQN agent with experience replay and target network
- `train.py` - Training script with TensorBoard logging
- `evaluate.py` - Evaluation script with statistics
- `export_onnx.py` - Export model to ONNX for web deployment

## Training Tips

1. **Start small**: Train for 1000 episodes first to verify setup
2. **Monitor TensorBoard**: Watch for loss stability and score improvement
3. **Tune epsilon decay**: Slower decay (higher value) = more exploration
4. **Use GPU**: Set `CUDA_VISIBLE_DEVICES=0` if available
5. **Checkpoint frequently**: Models can take hours to train

## Expected Performance

| Training Duration | Avg Score | Max Tile Rate (512+) |
|-------------------|-----------|----------------------|
| 10k episodes      | ~2,000    | ~20%                 |
| 50k episodes      | ~4,000    | ~40%                 |
| 100k+ episodes    | ~6,000+   | ~60%                 |

Random baseline: ~500 avg score, 64-128 max tile typical

## Hyperparameters

Default values work well, but you can tune:

```python
LEARNING_RATE = 1e-4      # Lower = more stable, slower learning
GAMMA = 0.99              # Discount factor
EPSILON_DECAY = 100000    # Steps for epsilon to reach minimum
BATCH_SIZE = 256          # Larger = more stable, more memory
BUFFER_SIZE = 100000      # Experience replay capacity
TARGET_UPDATE = 1000      # Target network update frequency
```
