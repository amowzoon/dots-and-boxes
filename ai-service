# ai-service/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class DotsAndBoxesAI:
    def __init__(self, depth=3):
        self.max_depth = depth
    
    def get_move(self, game_state):
        """Get the best move for the current game state"""
        possible_moves = self.get_possible_moves(game_state)
        
        if not possible_moves:
            return None
            
        best_score = float('-inf')
        best_move = possible_moves[0]
        
        for move in possible_moves:
            new_state = self.make_move(game_state.copy(), move)
            score = self.minimax(new_state, self.max_depth - 1, False, float('-inf'), float('inf'))
            
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move
    
    def minimax(self, game_state, depth, is_maximizing, alpha, beta):
        if depth == 0 or self.is_game_over(game_state):
            return self.evaluate_board(game_state)
            
        possible_moves = self.get_possible_moves(game_state)
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in possible_moves:
                new_state = self.make_move(game_state.copy(), move)
                eval = self.minimax(new_state, depth - 1, False, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in possible_moves:
                new_state = self.make_move(game_state.copy(), move)
                eval = self.minimax(new_state, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    
    def get_possible_moves(self, game_state):
        moves = []
        grid_size = game_state['gridSize']
        horizontal_lines = game_state['horizontalLines']
        vertical_lines = game_state['verticalLines']
        
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
    
    def make_move(self, game_state, move):
        # Create a deep copy to avoid modifying original
        new_state = {
            'gridSize': game_state['gridSize'],
            'horizontalLines': [row[:] for row in game_state['horizontalLines']],
            'verticalLines': [row[:] for row in game_state['verticalLines']],
            'boxes': [row[:] for row in game_state['boxes']],
            'currentPlayer': game_state['currentPlayer'],
            'scores': game_state['scores'].copy()
        }
        
        if move['type'] == 'horizontal':
            new_state['horizontalLines'][move['row']][move['col']] = True
        else:
            new_state['verticalLines'][move['row']][move['col']] = True
        
        # Check for completed boxes
        completed_boxes = self.check_completed_boxes(new_state, move)
        
        if completed_boxes:
            for box in completed_boxes:
                new_state['boxes'][box['row']][box['col']] = new_state['currentPlayer']
                new_state['scores'][new_state['currentPlayer'] - 1] += 1
            # Same player continues if boxes were completed
        else:
            # Switch players
            new_state['currentPlayer'] = 3 - new_state['currentPlayer']  # Switch between 1 and 2
        
        return new_state
    
    def check_completed_boxes(self, game_state, move):
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
    
    def evaluate_board(self, game_state):
        """Evaluate the board state for the AI player (player 2)"""
        player1_score = game_state['scores'][0]
        player2_score = game_state['scores'][1]
        
        # Simple evaluation: score difference
        return player2_score - player1_score
    
    def is_game_over(self, game_state):
        total_boxes = game_state['gridSize'] * game_state['gridSize']
        total_filled = sum(game_state['scores'])
        return total_filled == total_boxes

# Initialize AI with different difficulty levels
ai_instances = {
    'easy': DotsAndBoxesAI(depth=2),
    'medium': DotsAndBoxesAI(depth=3),
    'hard': DotsAndBoxesAI(depth=4)
}

@app.route('/api/ai/move', methods=['POST'])
def get_ai_move():
    try:
        data = request.json
        game_state = data['gameState']
        difficulty = data.get('difficulty', 'medium')
        
        ai = ai_instances.get(difficulty, ai_instances['medium'])
        move = ai.get_move(game_state)
        
        return jsonify({
            'success': True,
            'move': move
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
