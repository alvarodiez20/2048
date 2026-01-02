# 2048 - Rust + WebAssembly

[![CI](https://github.com/alvarodiez20/2048/actions/workflows/ci.yml/badge.svg)](https://github.com/alvarodiez20/2048/actions/workflows/ci.yml)
[![Deploy](https://github.com/alvarodiez20/2048/actions/workflows/deploy.yml/badge.svg)](https://github.com/alvarodiez20/2048/actions/workflows/deploy.yml)

**[🎮 Play the game online!](https://alvarodiez20.github.io/2048/)**

A complete 2048 game implementation featuring:
- **Pure Rust core engine** with deterministic, seedable PRNG
- **Native CLI** for interactive play and headless simulations
- **WebAssembly build** for browser-based gameplay
- **Canvas-based web UI** with modern, responsive design

```
┌─────────────────────────────────────────────────────────────┐
│                        2048 Repository                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────────────────┐  │
│   │   Core   │───▶│   CLI    │    │        Web UI        │  │
│   │  (Rust)  │    │  (Rust)  │    │   (TypeScript +      │  │
│   │          │    │          │    │    Canvas)           │  │
│   └──────────┘    └──────────┘    └──────────────────────┘  │
│        │                                    ▲               │
│        │                                    │               │
│        │          ┌──────────┐              │               │
│        └─────────▶│   WASM   │──────────────┘               │
│                   │ (Rust +  │                              │
│                   │  bindgen)│                              │
│                   └──────────┘                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Repository Structure

```
/2048
├── Cargo.toml              # Workspace root
├── README.md               # This file
├── LICENSE                 # MIT License
├── .gitignore
│
├── core/                   # Pure Rust game engine
│   ├── Cargo.toml
│   └── src/lib.rs
│
├── cli/                    # Native CLI runner
│   ├── Cargo.toml
│   └── src/main.rs
│
├── wasm/                   # WebAssembly bindings
│   ├── Cargo.toml
│   └── src/lib.rs
│
├── python/                 # Python bindings (PyO3)
│   ├── Cargo.toml
│   ├── pyproject.toml
│   └── src/lib.rs
│
├── rl/                     # Reinforcement learning training
│   ├── requirements.txt
│   ├── game_env.py
│   ├── dqn_agent.py
│   ├── train.py
│   ├── evaluate.py
│   └── export_onnx.py
│
├── web/                    # Static web application
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   └── src/
│       ├── main.ts
│       ├── ai-player.ts   # AI player integration
│       └── style.css
│
└── .github/
    └── workflows/
        ├── ci.yml          # GitHub Actions CI
        └── deploy.yml      # GitHub Pages deployment
```

## 🤖 AI Player Features

The web app includes an integrated AI player that can:
- **Watch AI Play**: Let the AI play automatically at adjustable speed
- **Get Hints**: Ask the AI for the best move suggestion
- **Random Mode**: Falls back to random moves if no trained model is available

### Training Your Own AI

```bash
# Setup
cd rl
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train (CPU, ~10k episodes)
python train.py --episodes 10000 --seed 42

# Evaluate
python evaluate.py --model checkpoints/best_model.pt --episodes 100

# Export for web deployment
python export_onnx.py --model checkpoints/best_model.pt \
    --output ../web/public/models/ai_model.onnx
```

See [rl/README.md](./rl/README.md) for detailed training instructions.

## 🚀 Quick Start

### Prerequisites

- **Rust** (stable): [Install Rust](https://rustup.rs/)
- **wasm-pack**: `cargo install wasm-pack`
- **Node.js** (v18+): [Install Node.js](https://nodejs.org/)

### Run CLI (Interactive Mode)

```bash
# Build and run
cargo run -p game-2048-cli

# Or with a specific seed
cargo run -p game-2048-cli -- --seed 12345
```

Controls: `W` `A` `S` `D` or Arrow keys | `R` to restart | `Q` to quit

### Run CLI (Headless Simulations)

```bash
# Run 100 episodes with random policy
cargo run -p game-2048-cli -- --episodes 100 --seed 42 --policy random

# With verbose output
cargo run -p game-2048-cli -- --episodes 10 --verbose
```

### Run Web UI (Development)

```bash
# Step 1: Build WASM module
cd wasm
wasm-pack build --target web --out-dir ../web/src/wasm-pkg

# Step 2: Start dev server
cd ../web
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

### Build Web UI (Production)

```bash
# Build WASM (if not already done)
cd wasm
wasm-pack build --target web --out-dir ../web/src/wasm-pkg

# Build for production
cd ../web
npm run build
```

Output will be in `web/dist/` - ready for static hosting.

## 🌐 Deploy to GitHub Pages

### Option 1: Manual Deployment

```bash
# Build everything
cd wasm && wasm-pack build --target web --out-dir ../web/src/wasm-pkg
cd ../web && npm install && npm run build

# Deploy dist folder to gh-pages branch
npx gh-pages -d dist
```

### Option 2: GitHub Actions (Automatic)

The included `.github/workflows/ci.yml` automatically builds and can be extended to deploy to GitHub Pages on push to `main`.

## 🎮 Game Rules

1. Tiles with the same number merge into one when they touch
2. After every move, a new tile appears (90% chance of 2, 10% chance of 4)
3. The game ends when no moves are possible
4. Goal: Reach the 2048 tile (and beyond!)

## 🧪 Testing

```bash
# Run core engine tests
cargo test -p game-2048-core

# Run all tests
cargo test

# With verbose output
cargo test -- --nocapture
```

## 🔧 Development

### Code Formatting

```bash
cargo fmt --all
```

### Linting

```bash
cargo clippy --all-targets --all-features
```

### Recommended Tool Versions

- Rust: stable (1.70+)
- wasm-pack: 0.12+
- Node.js: 18+
- npm: 9+

## 📖 API Reference

### Core Engine (`game-2048-core`)

```rust
use game_2048_core::{Game, Action, StepResult};

// Create a new game with seed
let mut game = Game::new(42);

// Execute a move
let result: StepResult = game.step(Action::Left);
println!("Changed: {}, Reward: {}, Done: {}", result.changed, result.reward, result.done);

// Query state
let board: &[u16; 16] = game.board();   // Row-major order
let score: u32 = game.score();
let is_over: bool = game.is_done();
let legal: [bool; 4] = game.legal_actions(); // [Up, Down, Left, Right]

// Reset with new seed
game.reset(123);
```

### WASM Module (`game-2048-wasm`)

```javascript
import init, { WasmGame } from './wasm-pkg/game_2048_wasm.js';

await init();

const game = new WasmGame(42n); // Note: BigInt for u64 seed

// Execute a move (0=Up, 1=Down, 2=Left, 3=Right)
const result = game.step(2); // Left
console.log(result); // { board: [...], score, reward, changed, done }

// Query state
const board = game.getBoard();     // Uint16Array
const score = game.getScore();     // number
const isDone = game.isDone();      // boolean
const legal = game.getLegalActions(); // Uint8Array [Up, Down, Left, Right]
```

## 🐍 Future Python RL Integration

The core engine is designed for easy integration with Python-based reinforcement learning:

### Approach 1: JSON over stdin/stdout

Use the CLI in headless mode and communicate via JSON:

```python
import subprocess
import json

# Run simulations
result = subprocess.run(
    ['./target/release/game-2048-cli', '--episodes', '1000', '--policy', 'random'],
    capture_output=True, text=True
)
# Parse output for statistics
```

For a custom RL loop, extend the CLI with a `--json` mode that outputs state after each step.

### Approach 2: Python Extension (Recommended)

Use [PyO3](https://pyo3.rs/) and [Maturin](https://github.com/PyO3/maturin) to create native Python bindings:

1. Add a new crate `/python` with PyO3 bindings
2. Expose `Game`, `Action`, `StepResult` to Python
3. Build with `maturin develop`

Example Python API (future):

```python
from game_2048 import Game, Action

env = Game(seed=42)

for _ in range(1000):
    action = your_policy(env.board)
    result = env.step(action)
    
    if result.done:
        env.reset(seed=new_seed)
```

### Key Design Decisions for RL

- **Deterministic**: Same seed + action sequence = same game
- **Minimal API**: `step(action)` returns everything needed
- **Flat state**: Board is a simple `[u16; 16]` array
- **Efficient**: No allocations in the hot path
- **Reward signal**: Returns merge points as immediate reward

## 📄 License

MIT License - see [LICENSE](./LICENSE) for details.