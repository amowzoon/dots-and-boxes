# Dots and Boxes - Deep Reinforcement Learning

A deep reinforcement learning implementation of the classic Dots and Boxes game using AlphaZero-style training with Monte Carlo Tree Search (MCTS) and neural networks.

---

## Table of Contents
- [Overview](#overview)
- [Game Description](#game-description)
- [Our Approach](#our-approach)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [References](#references)

---

## Overview

This project implements an AI agent that learns to play Dots and Boxes through self-play reinforcement learning. Inspired by DeepMind's AlphaZero, our system combines:

- **Monte Carlo Tree Search (MCTS)** for strategic planning
- **Deep Convolutional Neural Networks** for board evaluation and move prediction
- **Self-play training** for continuous improvement
- **Web-based interface** for human vs. AI gameplay

The trained models achieve strong performance on 2×2 and 3×3 boards, demonstrating strategic understanding of the game's complexities including chain captures and sacrifice strategies.

---

## Game Description

### Rules
1. **Setup**: Players start with a rectangular grid of dots
2. **Gameplay**: Players alternate drawing horizontal or vertical lines between adjacent dots
3. **Scoring**: When a player completes the fourth side of a 1×1 box, they:
   - Claim that box and score one point
   - Immediately take another turn
4. **Victory**: The player who claims the most boxes when the grid is full wins

### Strategic Depth
Despite simple rules, Dots and Boxes exhibits surprising complexity:
- **Chain Captures**: Completing one box often enables capturing multiple boxes in sequence
- **Sacrifice Strategy**: Intentionally giving away boxes to minimize opponent's chain length
- **Parity & Endgame**: Total box count creates critical endgame patterns
- **Double-Cross**: Advanced technique for controlling long chains

This complexity makes it an excellent testbed for AI game-playing algorithms.

---

## Our Approach

### Board Representation

We use a tensor-based representation inspired by AlphaZero:

**Game State Tensor**: Shape `(n, n, 5)` where:
- **Channels 0-3**: Binary indicators for each box's edges (top, bottom, left, right)
- **Channel 4**: Player ownership (0 = unclaimed, 1 = Player A, -1 = Player B)

**Additional State Tracking**:
- `boxes_playerA`, `boxes_playerB`: Lists of claimed box coordinates
- `recent_edge`: Most recently drawn edge
- `On_Offensive`: Boolean indicating if current player gets another turn

This representation enables:
- Efficient edge placement with automatic neighbor updates
- Clear ownership tracking for both players
- Natural input format for convolutional neural networks

### Neural Network Architecture

Our model uses a residual CNN architecture with dual outputs:

```
Input (n×n×5 tensor)
    ↓
3×3 Conv + ReLU (initial feature extraction)
    ↓
3× Residual Blocks (feature learning)
    ↓
    ├─→ Policy Head: 1×1 Conv → Flatten → FC → Action Probabilities
    └─→ Value Head: 1×1 Conv → Flatten → FC(64) → FC(1) → tanh → Win Probability
```

**Model Details**:
- **Input**: 5-channel board state tensor
- **Residual Blocks**: 3 blocks with skip connections for gradient flow
- **Policy Output**: Probability distribution over all possible moves
- **Value Output**: Estimated win probability ∈ [-1, 1]
- **Parameters**: ~57,840 (2×2) / ~58,508 (3×3)

**Loss Function**:
```
L_total = L_policy + L_value
L_policy = -Σ(π_target · log(softmax(p_pred)))
L_value = (1/N)Σ(v_pred - z_target)²
```

### Monte Carlo Tree Search (MCTS)

MCTS guides exploration during both training and inference through four phases:

1. **Selection**: Traverse tree using Upper Confidence Bound (UCB) to balance exploration/exploitation
2. **Expansion**: Add new child node for unexplored action
3. **Simulation**: Use neural network to evaluate position (no random rollouts)
4. **Backpropagation**: Update visit counts and values up the tree

The neural network provides both move probabilities (to guide selection) and position evaluation (to avoid full simulations).

### Training Pipeline

**Self-Play Data Generation**:
1. Initialize empty board and MCTS with current model
2. For each move:
   - Run MCTS to generate improved policy π
   - Select action stochastically from π distribution
   - Record tuple: (board_state, π, current_player)
3. When game ends, append outcome z to all tuples: (s, π, z, player_id)

**Training Cycle**:
```
Loop until convergence or until max_training_cycles is reached:
    1. Generate 300+ self-play games
    2. Train neural network for 50 epochs on collected data
    3. Every 10 cycles: Evaluate against pure MCTS baseline
    4. If new model wins 100% → Update best model
```

**Data Augmentation** (planned):
- Board rotations and reflections to increase effective dataset size

---

### Key Components

#### 1. `Dots_and_Boxes_Model.ipynb`
- Complete game implementation with tensor-based state representation
- Neural network definition (ResNet architecture)
- MCTS implementation with neural network guidance
- Self-play data generation and training loops
- Model evaluation against pure MCTS baseline

#### 2. `notebook_to_api.py` (Flask API)
- Loads notebook and skips training cells
- Loads trained models for different board sizes
- Bridges notebook code with web interface

#### 3. `index.html` (Web Interface)
- Interactive game board with click-to-draw mechanics
- Real-time score tracking for both players
- Difficulty selection (Easy, Medium, Hard) via MCTS simulation count
- Grid size customization (2×2 to 10×10)
- Visual distinction between player and AI moves

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- PyTorch (with CUDA support recommended for training)
- Flask
- NumPy

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/amowzoon/dots-and-boxes.git
   cd dots-and-boxes
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision flask numpy jupyter
   ```

3. **Download or train models**:
   - Pre-trained models: `best_model_2x2.pth`, `best_model_3x3.pth` should be in root directory
   - To train from scratch: Open and run `Dots_and_Boxes_Model.ipynb`

---

## Usage

### Training a New Model

1. Open `Dots_and_Boxes_Model.ipynb` in Jupyter/Colab
2. Configure hyperparameters:
   ```python
   BOARD_SIZE = 2  # or 3, 4, etc.
   NUM_MCTS_SIMS = 50
   NUM_TRAINING_CYCLES = 50
   GAMES_PER_CYCLE = 300
   ```
3. Run all cells to begin self-play training
4. Trained model saves as `best_model_{size}x{size}.pth`
5. Average losses and win rates are written to a file "losses.txt" and "win_rates.txt" in the same documentation
   * The file will be overwritten every time trainer.trian() is run, please rename your files after running training to save losses
6. To plot losses ensure that the file name for loses is the same as "loses_{n}x{n}.txt" or edit the code where marked to match file names

**Training Resources Used**:
- Google Colab T4 GPU
- BU SCC A40/L40 GPUs (4 cores)
- Local Ryzen 9 CPU

### Playing Against the AI

1. **Start the Flask server**:
   ```bash
   python ai-service.py
   ```
   
   You should see:
   ```
   Loading notebook: Dots_and_Boxes_Model.ipynb
   Notebook loaded successfully
   Game class found
   MCTS class found
   Loaded best_model_2x2.pth for 2x2 board
   Loaded best_model_3x3.pth for 3x3 board
   * Running on http://localhost:5000
   ```

2. **Open the web interface**:
   ```bash
   # Open in your browser:
   http://localhost:5000
   ```

3. **Configure game settings**:
   - Select board size (2×2 to 6×6)
   - Game automatically starts
   - Can select "New Game" to reset or "Play Again" if game commpleted

4. **Play**:
   - Click between dots to draw lines
   - Your moves appear in one color, AI moves in another
   - Completed boxes show player ownership
   - Game automatically handles chain captures (bonus turns)

---

## References

1. **Zhang, Y., Li, S., & Xiong, X.** (2019). "A Study on the Game System of Dots and Boxes Based on Reinforcement Learning." *Chinese Control And Decision Conference (CCDC)*.

2. **Silver, D., et al.** (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm." *arXiv:1712.01815*.

3. **Silver, D., Huang, A., Maddison, C.J., et al.** (2016). "Mastering the Game of Go with Deep Neural Networks and Tree Search." *Nature*, 529, 484–489.

4. **Mnih, V., Kavukcuoglu, K., Silver, D., et al.** (2013). "Playing Atari with Deep Reinforcement Learning." *arXiv:1312.5602*.

5. **Josh Varty's AlphaZeroSimple**: https://github.com/JoshVarty/AlphaZeroSimple

6. **Flask Documentation**: https://flask.palletsprojects.com
