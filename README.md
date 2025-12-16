# Asalto Game AI (Team 20)

This project implements an AI bot for the Asalto board game using Minimax algorithm with Alpha-Beta pruning and Heuristic evaluation. It also includes an experimental Deep Reinforcement Learning training script.

## Project Structure

- `Asalto.py`: The game engine and testing framework.
- `Team20.py`: The main entry point for our AI player.
- `RebelAI.py`: Logic for the Rebel player (Minimax + Heuristic).
- `OfficerAI.py`: Logic for the Officer player (Minimax + Heuristic).
- `TeamDQN.py`: Player implementation using the trained Neural Network.
- `AsaltoTest.py`: Script to run matches between different AI models (e.g., Minimax vs DQN).
- `training/`: Directory containing training scripts and model definitions.

## How to Run

### 1. Setup Environment

First, create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Game Simulation (Minimax AI)

To watch the AI play against itself using the Minimax algorithm:

```bash
python3 Asalto.py
```

This will start a game where Team20 plays both Rebels and Officers. The board state will be printed after each round.

### 3. Train Neural Network (Experimental)

To train a Deep Q-Network (DQN) style model:

```bash
cd training
python3 train.py
```

**Training Strategy**: The training script now uses **Minimax Guidance**. The Rebel player uses the Minimax algorithm (100% probability) to generate high-quality moves, forcing the Neural Network (playing as Officer) to learn how to defeat a strong opponent.

### 4. Compare Models (Minimax vs DQN)

To run a head-to-head comparison between the Minimax algorithm and the trained DQN model:

```bash
python3 AsaltoTest.py
```

This script runs two matches (swapping roles) and outputs the final winner based on the results.

## Algorithms Implemented

- **Rebel Strategy**: 
    - Uses **Minimax** algorithm with **Alpha-Beta Pruning**.
    - Default search depth is **4**.
    - Supports **Iterative Deepening** (optional) to maximize search depth within the 10-second time limit.
    - Heuristic evaluation focuses on fortress occupation, formation cohesion, and trapping officers.

- **Officer Strategy**: 
    - Uses **Minimax** algorithm with **Alpha-Beta Pruning**.
    - Default search depth is **4**.
    - Supports **Iterative Deepening** (optional).
    - Prioritizes capturing rebels (mandatory rule), defending the fortress entrance, and maintaining mobility.

### Configuration

You can toggle **Iterative Deepening** in `Team20.py`:

```python
# Global configuration
USE_ITERATIVE_DEEPENING = False # Set to True to enable dynamic depth
```

When enabled, the AI will progressively search deeper (Depth 1 -> 2 -> 3...) until the time limit (9.0s) is reached, ensuring optimal play without timeouts.

## Strategy Analysis

### 1. Current Strategy: Minimax + Alpha-Beta + Iterative Deepening
We selected this as our **final submission strategy** because:
- **Reliability**: It guarantees finding the best move within the search horizon (Depth 4).
- **Efficiency**: Alpha-Beta pruning significantly reduces the search space, allowing deeper searches within the 10s limit.
- **Safety**: Iterative Deepening ensures we always have a valid move ready, eliminating timeout risks.

### 2. Experimental Strategy: Deep Q-Network (DQN)
We implemented a DQN model trained via **Imitation Learning** (guided by Minimax).
- **Result**: The trained model performed weaker than the pure Minimax algorithm.
- **Analysis**: Asalto is a perfect information game where precise calculation (Search) often beats approximation (Neural Network), especially given limited training time. The network struggled to learn the precise "mandatory capture" rules and complex fortress occupation tactics.

### 3. Alternative Strategies (Future Work)
We identified other potential architectures that could be effective:

- **Q-Learning + Heuristic Pruning (Rebel Focus)**
    - *Concept*: Combine Reinforcement Learning with rule-based pruning to filter out bad moves.
    - *Advantage*: Good for the **Early Game** where Rebels need to coordinate mass movement towards the fortress.

- **MCTS + Dual Network (Officer Focus)**
    - *Concept*: AlphaZero-style architecture using Monte Carlo Tree Search guided by Policy/Value networks.
    - *Advantage*: Excellent for **Mid/Late Game** precision, helping Officers find complex capture chains to reduce Rebel count below 9.

- **Multi-Piece Activation + Pruning**
    - *Concept*: Optimization technique to only search moves for "active" pieces (e.g., those near opponents).
    - *Advantage*: Reduces branching factor, lowering timeout risk.

## Compliance

The core submission files (`Team20.py`, `RebelAI.py`, `OfficerAI.py`) rely **exclusively on Python Standard Library** modules (`random`, `copy`, `math`, `time`). This ensures maximum compatibility and stability, strictly adhering to the assignment requirements.

The experimental Deep Learning components (`TeamDQN.py`, `training_minimax_guided/`) utilize `torch` and `numpy`, which are permitted as per `requirements.txt`.