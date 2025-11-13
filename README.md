# Dots and Boxes Algorithm

## Table of Contents
- [Overview](#overview)
- [Game Description](#game-description)
- [Project Components](#project-components)
- [Theoretical Background](#theoretical-background)
- [Implementation Details](#implementation-details)
- [References](#references)

---

## Overview

This project explores the classic game **Dots and Boxes** through multiple computational approaches, ranging from traditional game tree search algorithms to modern deep reinforcement learning techniques. The work demonstrates how artificial intelligence can learn to master strategic board games through sufficient training and coding practices.

**Dots and Boxes** is a pencil-and-paper game for two players, typically played on a rectangular grid of dots. Despite its simple rules, the game exhibits surprising strategic depth, making it an excellent testbed for game-playing AI algorithms. Not only must players be able to locate and fill in a box, but also players must be able to put themselves in a winning scenario through mapping out their own clusters whilst sacrificing clusters worth less points.

---

## Game Description

### Rules

1. **Setup**: Players start with a grid of dots
2. **Gameplay**: Players take turns drawing horizontal or vertical lines between adjacent dots
3. **Scoring**: When a player completes the fourth side of a box (1×1 square), they:
   - Claim that box
   - Score one point
   - Take another turn immediately
4. **Victory**: The player with the most boxes when all boxes are claimed wins

### Strategic Elements

- **Chain Captures**: Completing one box often sets up more boxes to capture
- **Sacrifice Strategy**: Sometimes giving away boxes minimizes opponent's chain length
- **Parity**: The total number of boxes and game structure create endgame patterns
- **Double-Cross Strategy**: Advanced technique for controlling long chains

The game is deceptively complex: while children can play it, optimal play requires deep lookahead and strategic planning similar to games like Go or Chess.

---

## Project Components

### 1. Jupyter Notebook: `Dots_and_Boxes_Experimentation.ipynb`

This Google Colab notebook contains the foundational implementation and experimentation framework:

---

### 2. Web-Based Implementation

A complete web application featuring:

#### **Frontend: `index.html`**
- Interactive UI
- Real-time score tracking
- Visual feedback via colored lines and boxes distinguishing players from one another
- User goes against an AI agent
- Difficulty selection (Easy, Medium, Hard)
- Grid size customization (2×2 to 10×10)

#### **Backend: `ai-service.py`**
- Flask REST API for AI moves
- Monte Carlo Tree Search (MCTS) implementation
- Greedy box completion check
- Configurable simulation limits

**"The best way to predict the future is to invent it."** - Alan Kay

This project represents a journey through decades of AI research, from minimax to MCTS to deep reinforcement learning, all applied to the humble game of Dots and Boxes.
