# Dots and Boxes Algorithm

## Table of Contents
- [Overview](#overview)
- [Game Description](#game-description)
- [Project Components](#project-components)

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

Representation 1 uses arrays to represent the edges and completed boxes: 
**edges**: Array of all edges in format (i, j, k) where (i, j, 0) is a line from dot (i, j) to dot (i, j+1) and (i, j, 1) is a line from (i, j) to (i+1, j)
**boxes**: Array of completed boxes in coordinates in form (i, j) 

There are two functions, one that initializes these arrays and the other that is called to set an edge and check if a box has been completed and it updates the arrays accordingly. 
The problems we had with this representation is that it wasn't intuitively clear what happened visually when you set down an edge and it would be hard to implement the 5 layers of representation in this so we moved to representation 2. 

Representation 2 is based on the representation used in the original paper where there is a tensor nxn elements to represent each box and each element has 5 channels to reprensent whether it's 4 sides have been filled as well as ownership. The other four layers that make up the game state are also implemented in this and are initialized in the class init function. 
To make an edge the user enters the box coordinates that they want to draw on (r, c) as well as what edge they want to draw (top, bottom, right or left). 

The function init instantiates 4 variables:
**board_tensor**: shape (n, n, 5) with 4 edges per box and an extra channel for player ownership
                  channels are in [top, bottom, left, right, player owner]
**boxes_playerA**: list of boxes owned by player A in (x,y)
**boxes_playerB**: list of boxes owned by player B
**recent_edge**: most recent edge drawn
**On_Offensive**: boolean variable representing whether a player has just completed a box and is going again 

The function **set_edge** is called to make a new edge and updates the board_tensor the box the edge is drawn in and automatically updates that edge for any neighbouring boxes that share it. It also updates *recent_edge* 

The function **check_box** checks whether an edge completes any boxes and returns the number of boxes completed. It also updates *boxes_playerA* or *boxes_playerB* as well as *On_Offensive* 

The function **print_board_ascii** prints the current board with all drawn edges and displays the coordinates of each box unless it is owned in which case it displays the player that owns it. It also displays what player's turn it is and what move in the game it is.  

The function **player_turn** takes in the inputs (r, c, edge_type, player, move) and calls the three previous functions above. 

The function **gameplay** uses the class definition to instantiate an actual game where users take turn adding edges to the game. It displays the winnner after all boxes have been filled. 

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
