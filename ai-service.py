from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import math
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


class MCTSNode:
    """Node in the MCTS tree"""
    
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move  # Move that led to this node
        
        self.children = []
        self.untried_moves = self.get_legal_moves()
        
        self.visits = 0
        self.wins = 0.0
        
    def get_legal_moves(self):
        """Get all legal moves from current state"""
        moves = []
        grid_size = self.game_state['gridSize']
        horizontal_lines = self.game_state['horizontalLines']
        vertical_lines = self.game_state['verticalLines']
        
        # Horizontal lines
        for row in range(len(horizontal_lines)):
            for col in range(len(horizontal_lines[row])):
                if not horizontal_lines[row][col]:
                    moves.append({'type': 'horizontal', 'row': row, 'col': col})
        
        # Vertical lines
        for row in range(len(vertical_lines)):
            for col in range(len(vertical_lines[row])):
                if not vertical_lines[row][col]:
                    moves.append({'type': 'vertical', 'row': row, 'col': col})
        
        return moves
    
    def is_fully_expanded(self):
        """Check if all children have been created"""
        return len(self.untried_moves) == 0
    
    def is_terminal(self):
        """Check if game is over"""
        total_boxes = self.game_state['gridSize'] ** 2
        total_filled = sum(self.game_state['scores'])
        return total_filled == total_boxes
    
    def best_child(self, exploration_weight=1.41):
        """
        Select best child using UCB1 formula
        
        UCB1 = (wins / visits) + C * sqrt(ln(parent_visits) / visits)
                exploitation           exploration
        """
        choices_weights = []
        
        for child in self.children:
            if child.visits == 0:
                # Unvisited nodes get infinite priority
                ucb1_score = float('inf')
            else:
                # UCB1 formula
                exploitation = child.wins / child.visits
                exploration = math.sqrt(math.log(self.visits) / child.visits)
                ucb1_score = exploitation + exploration_weight * exploration
            
            choices_weights.append(ucb1_score)
        
        # Return child with highest UCB1 score
        return self.children[choices_weights.index(max(choices_weights))]
    
    def most_visited_child(self):
        """Return the child with the most visits (used for final move selection)"""
        return max(self.children, key=lambda c: c.visits)
    
    def expand(self):
        """Add a new child node for an untried move"""
        move = self.untried_moves.pop()
        new_state = self.apply_move(self.deep_copy_state(), move)
        child_node = MCTSNode(new_state, parent=self, move=move)
        self.children.append(child_node)
        return child_node
    
    def deep_copy_state(self):
        """Create a deep copy of game state"""
        state = self.game_state
        return {
            'gridSize': state['gridSize'],
            'horizontalLines': [row[:] for row in state['horizontalLines']],
            'verticalLines': [row[:] for row in state['verticalLines']],
            'boxes': [row[:] for row in state['boxes']],
            'currentPlayer': state['currentPlayer'],
            'scores': state['scores'][:]
        }
    
    def apply_move(self, game_state, move):
        """Apply a move to the game state"""
        if move['type'] == 'horizontal':
            game_state['horizontalLines'][move['row']][move['col']] = True
        else:
            game_state['verticalLines'][move['row']][move['col']] = True
        
        # Check for completed boxes
        completed_boxes = self.check_completed_boxes(game_state, move)
        
        if completed_boxes:
            for box in completed_boxes:
                game_state['boxes'][box['row']][box['col']] = game_state['currentPlayer']
                game_state['scores'][game_state['currentPlayer'] - 1] += 1
            # Same player continues
        else:
            # Switch players
            game_state['currentPlayer'] = 3 - game_state['currentPlayer']
        
        return game_state
    
    def check_completed_boxes(self, game_state, move):
        """Check if move completes any boxes"""
        completed_boxes = []
        grid_size = game_state['gridSize']
        horizontal_lines = game_state['horizontalLines']
        vertical_lines = game_state['verticalLines']
        
        if move['type'] == 'horizontal':
            # Check box above
            if move['row'] > 0:
                if (horizontal_lines[move['row'] - 1][move['col']] and
                    vertical_lines[move['row'] - 1][move['col']] and
                    vertical_lines[move['row'] - 1][move['col'] + 1]):
                    completed_boxes.append({'row': move['row'] - 1, 'col': move['col']})
            
            # Check box below
            if move['row'] < grid_size:
                if (horizontal_lines[move['row'] + 1][move['col']] and
                    vertical_lines[move['row']][move['col']] and
                    vertical_lines[move['row']][move['col'] + 1]):
                    completed_boxes.append({'row': move['row'], 'col': move['col']})
        
        else:  # vertical move
            # Check box to the left
            if move['col'] > 0:
                if (vertical_lines[move['row']][move['col'] - 1] and
                    horizontal_lines[move['row']][move['col'] - 1] and
                    horizontal_lines[move['row'] + 1][move['col'] - 1]):
                    completed_boxes.append({'row': move['row'], 'col': move['col'] - 1})
            
            # Check box to the right
            if move['col'] < grid_size:
                if (vertical_lines[move['row']][move['col'] + 1] and
                    horizontal_lines[move['row']][move['col']] and
                    horizontal_lines[move['row'] + 1][move['col']]):
                    completed_boxes.append({'row': move['row'], 'col': move['col']})
        
        return completed_boxes
    
    def simulate(self):
        """
        Simulate a game from this node until the end (ROLLOUT)
        Uses a semi-intelligent policy: prioritize moves that complete boxes
        """
        state = self.deep_copy_state()
        
        # Play moves until game is over
        while True:
            total_boxes = state['gridSize'] ** 2
            total_filled = sum(state['scores'])
            
            if total_filled == total_boxes:
                # Game over, return result from AI's perspective (player 2)
                ai_score = state['scores'][1]
                player_score = state['scores'][0]
                
                if ai_score > player_score:
                    return 1.0  # AI wins
                elif ai_score < player_score:
                    return 0.0  # AI loses
                else:
                    return 0.5  # Tie
            
            # Get legal moves and categorize them
            legal_moves = []
            completing_moves = []  # Moves that complete a box
            safe_moves = []  # Moves that don't give opponent a box
            
            for row in range(len(state['horizontalLines'])):
                for col in range(len(state['horizontalLines'][row])):
                    if not state['horizontalLines'][row][col]:
                        move = {'type': 'horizontal', 'row': row, 'col': col}
                        legal_moves.append(move)
                        
                        # Check if this move completes a box
                        if self.move_completes_box(state, move):
                            completing_moves.append(move)
                        elif not self.move_gives_box(state, move):
                            safe_moves.append(move)
            
            for row in range(len(state['verticalLines'])):
                for col in range(len(state['verticalLines'][row])):
                    if not state['verticalLines'][row][col]:
                        move = {'type': 'vertical', 'row': row, 'col': col}
                        legal_moves.append(move)
                        
                        # Check if this move completes a box
                        if self.move_completes_box(state, move):
                            completing_moves.append(move)
                        elif not self.move_gives_box(state, move):
                            safe_moves.append(move)
            
            if not legal_moves:
                break
            
            # Smart move selection (biased random policy):
            # 1. Always take moves that complete boxes (80% of the time)
            # 2. Otherwise prefer safe moves (60% of the time)
            # 3. Sometimes make random moves (for exploration)
            
            if completing_moves and random.random() < 0.8:
                # Prioritize completing boxes
                move = random.choice(completing_moves)
            elif safe_moves and random.random() < 0.6:
                # Prefer safe moves
                move = random.choice(safe_moves)
            else:
                # Random move
                move = random.choice(legal_moves)
            
            state = self.apply_move(state, move)
    
    def move_completes_box(self, state, move):
        """Check if a move would complete at least one box"""
        completed = self.check_completed_boxes(state, move)
        return len(completed) > 0
    
    def move_gives_box(self, state, move):
        """
        Check if a move would allow opponent to complete a box on their next turn
        (i.e., creates a box with 3 sides completed)
        """
        # Temporarily apply the move
        temp_state = self.deep_copy_state_from(state)
        if move['type'] == 'horizontal':
            temp_state['horizontalLines'][move['row']][move['col']] = True
        else:
            temp_state['verticalLines'][move['row']][move['col']] = True
        
        # Check all boxes to see if any now have exactly 3 sides
        grid_size = temp_state['gridSize']
        
        for box_row in range(grid_size):
            for box_col in range(grid_size):
                sides = 0
                
                # Count completed sides
                if temp_state['horizontalLines'][box_row][box_col]:
                    sides += 1
                if temp_state['horizontalLines'][box_row + 1][box_col]:
                    sides += 1
                if temp_state['verticalLines'][box_row][box_col]:
                    sides += 1
                if temp_state['verticalLines'][box_row][box_col + 1]:
                    sides += 1
                
                # If exactly 3 sides, opponent can complete this box
                if sides == 3:
                    return True
        
        return False
    
    def deep_copy_state_from(self, state):
        """Create a deep copy of a given game state"""
        return {
            'gridSize': state['gridSize'],
            'horizontalLines': [row[:] for row in state['horizontalLines']],
            'verticalLines': [row[:] for row in state['verticalLines']],
            'boxes': [row[:] for row in state['boxes']],
            'currentPlayer': state['currentPlayer'],
            'scores': state['scores'][:]
        }
    
    def backpropagate(self, result):
        """Update node statistics up the tree"""
        node = self
        
        while node is not None:
            node.visits += 1
            
            # Add win from perspective of player who made the move leading to this node
            # If this node's move was made by player 2 (AI), and AI won, add the win
            if node.move is not None:
                # Check which player made this move
                # After a move, currentPlayer is either the same (box completed) or switched
                # We need to track who made the move, not who's turn it is now
                move_player = 3 - node.game_state['currentPlayer']  # The player who just moved
                
                # If the move was made by AI (player 2), reward based on result
                if move_player == 2:
                    node.wins += result
                else:
                    # If move was made by human (player 1), reward is inverted
                    node.wins += (1.0 - result)
            
            node = node.parent


class DotsAndBoxesMCTS:
    """MCTS-based AI for Dots and Boxes"""
    
    def __init__(self, time_limit=1.0, simulation_limit=None):
        """
        Initialize MCTS AI
        
        Args:
            time_limit: Maximum time (seconds) to search
            simulation_limit: Maximum number of simulations (if None, use time_limit)
        """
        self.time_limit = time_limit
        self.simulation_limit = simulation_limit
    
    def get_move(self, game_state):
        """
        Get the best move using MCTS
        
        This is the main MCTS loop with 4 phases:
        1. SELECT - Navigate down tree using UCB
        2. EXPAND - Add new child node
        3. SIMULATE - Play random game to end (rollout)
        4. BACKPROPAGATE - Update statistics
        """
        root = MCTSNode(game_state)
        
        # CRITICAL FIX: Always take moves that complete boxes immediately!
        # This is a greedy check before MCTS even runs
        completing_moves = []
        for move in root.untried_moves:
            if self.move_completes_box(root, move):
                completing_moves.append(move)
        
        if completing_moves:
            # If there are box-completing moves, take the best one
            print(f"🎯 Found {len(completing_moves)} box-completing moves! Taking one immediately.")
            return completing_moves[0]
        
        # If only one legal move, return it immediately
        if len(root.untried_moves) == 1:
            return root.untried_moves[0]
        
        start_time = time.time()
        simulations = 0
        
        # Run MCTS simulations
        while True:
            # Check stopping conditions
            if self.simulation_limit and simulations >= self.simulation_limit:
                break
            if time.time() - start_time >= self.time_limit:
                break
            
            node = root
            
            # PHASE 1: SELECTION
            # Navigate down the tree using UCB until we find a node to expand
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child()
            
            # PHASE 2: EXPANSION
            # Add a new child node if possible
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            # PHASE 3: SIMULATION (ROLLOUT)
            # Play random moves until the game ends
            result = node.simulate()
            
            # PHASE 4: BACKPROPAGATION
            # Update statistics up the tree
            node.backpropagate(result)
            
            simulations += 1
        
        # Return the move with the most visits (most explored)
        if not root.children:
            # Fallback to random if no simulations completed
            return random.choice(root.untried_moves) if root.untried_moves else None
        
        # Choose best child based on combination of visits and win rate
        # This helps avoid moves that were visited a lot but have poor win rates
        best_child = None
        best_score = -1
        
        for child in root.children:
            if child.visits == 0:
                continue
            
            win_rate = child.wins / child.visits
            visit_weight = child.visits / root.visits
            
            # Combined score: weighted average of win rate and visit proportion
            # High visits + high win rate = best move
            score = 0.7 * win_rate + 0.3 * visit_weight
            
            if score > best_score:
                best_score = score
                best_child = child
        
        if best_child is None:
            best_child = root.most_visited_child()
        
        # Debug info
        print(f"MCTS completed {simulations} simulations in {time.time() - start_time:.2f}s")
        print(f"Best move: {best_child.move} (visits: {best_child.visits}, win rate: {best_child.wins/best_child.visits:.2%})")
        
        return best_child.move
    
    def move_completes_box(self, node, move):
        """Check if a move would complete at least one box (helper for DotsAndBoxesMCTS)"""
        completed = node.check_completed_boxes(node.game_state, move)
        return len(completed) > 0


# Initialize AI instances with different difficulty levels
ai_instances = {
    'easy': DotsAndBoxesMCTS(time_limit=0.5, simulation_limit=100),      # Fast, few simulations
    'medium': DotsAndBoxesMCTS(time_limit=1.0, simulation_limit=500),    # Moderate
    'hard': DotsAndBoxesMCTS(time_limit=2.0, simulation_limit=2000)      # Slow, many simulations
}


@app.route('/api/ai/move', methods=['POST'])
def get_ai_move():
    try:
        data = request.json
        game_state = data['gameState']
        difficulty = data.get('difficulty', 'medium')
        
        # Get appropriate AI instance
        ai = ai_instances.get(difficulty, ai_instances['medium'])
        
        # Get move using MCTS
        move = ai.get_move(game_state)
        
        if move is None:
            return jsonify({
                'success': False,
                'error': 'No valid moves available'
            }), 400
        
        return jsonify({
            'success': True,
            'move': move
        })
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'algorithm': 'MCTS (Monte Carlo Tree Search)',
        'features': [
            'Adaptive tree search',
            'Random rollout simulations',
            'UCB1 selection',
            'No evaluation function needed'
        ]
    })


@app.route('/', methods=['GET'])
def index():
    return """
    <h1>Dots and Boxes AI - MCTS Version</h1>
    <p><strong>Algorithm:</strong> Monte Carlo Tree Search (MCTS)</p>
    <h2>How it works:</h2>
    <ol>
        <li><strong>Selection:</strong> Navigate tree using UCB1 formula</li>
        <li><strong>Expansion:</strong> Add new unexplored node</li>
        <li><strong>Simulation:</strong> Play random moves to game end</li>
        <li><strong>Backpropagation:</strong> Update win/visit statistics</li>
    </ol>
    <h2>Endpoints:</h2>
    <ul>
        <li>POST /api/ai/move - Get AI move</li>
        <li>GET /api/health - Health check</li>
    </ul>
    <h2>Difficulty Levels:</h2>
    <ul>
        <li><strong>Easy:</strong> 100 simulations, 0.5s limit</li>
        <li><strong>Medium:</strong> 500 simulations, 1.0s limit</li>
        <li><strong>Hard:</strong> 2000 simulations, 2.0s limit</li>
    </ul>
    """


if __name__ == '__main__':
    print("=" * 60)
    print("Starting Dots and Boxes AI Server (MCTS Version)")
    print("=" * 60)
    print("Algorithm: Monte Carlo Tree Search")
    print("Key Features:")
    print("  ✓ Adaptive tree search (focuses on promising moves)")
    print("  ✓ Random rollout simulations (no evaluation function)")
    print("  ✓ UCB1 selection (balances exploration/exploitation)")
    print("  ✓ Win-rate based decision making")
    print("=" * 60)
    print("Server running at http://localhost:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
