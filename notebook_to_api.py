"""
Dynamic Notebook-to-API Flask Server
Loads your Jupyter notebook code and exposes it as a REST API
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from functools import wraps
import json
import os
import sys
import nbformat
from nbformat import v4 as nbf
import random

app = Flask(__name__)
CORS(app)

# Load API configuration
def load_config():
    """Load API configuration from file or environment"""
    config = {
        'api_key': os.environ.get('API_KEY'),
        'notebook_path': os.environ.get('NOTEBOOK_PATH', 'dots_and_boxes.ipynb'),
        'port': int(os.environ.get('PORT', 5000)),
        'host': os.environ.get('HOST', '0.0.0.0')
    }
    
    # Try to load from config file if exists
    if os.path.exists('api_config.json'):
        try:
            with open('api_config.json', 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            print(f"Warning: Could not load api_config.json: {e}")
    
    return config

CONFIG = load_config()

# API Key Authentication Decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # In development, allow requests without API key
        if os.environ.get('FLASK_ENV') == 'development':
            return f(*args, **kwargs)
            
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        if CONFIG.get('api_key') and api_key != CONFIG['api_key']:
            return jsonify({'error': 'Invalid API key'}), 403
            
        return f(*args, **kwargs)
    return decorated_function


def load_notebook_code(notebook_path):
    """
    Load and execute code from Jupyter notebook
    Returns the notebook's global namespace
    """
    print(f"Loading notebook: {notebook_path}")
    
    if not os.path.exists(notebook_path):
        print(f"ERROR: Notebook not found at {notebook_path}")
        print("Available files:", os.listdir('.'))
        return None
    
    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Create namespace for execution
        notebook_globals = {}
        
        # Execute each code cell
        for cell in nb.cells:
            if cell.cell_type == 'code':
                try:
                    # Skip cells that might cause issues (like magic commands)
                    code = cell.source
                    if code.strip().startswith('%') or code.strip().startswith('!'):
                        continue
                    
                    # Execute the cell
                    exec(code, notebook_globals)
                except Exception as e:
                    print(f"Warning: Error executing cell: {e}")
                    continue
        
        print("Notebook loaded successfully!")
        print("Available classes:", [k for k in notebook_globals.keys() if isinstance(notebook_globals[k], type)])
        
        return notebook_globals
        
    except Exception as e:
        print(f"ERROR loading notebook: {e}")
        import traceback
        traceback.print_exc()
        return None


# Load notebook on startup
NOTEBOOK_NAMESPACE = None

def get_notebook_namespace():
    """Get or reload notebook namespace"""
    global NOTEBOOK_NAMESPACE
    
    if NOTEBOOK_NAMESPACE is None:
        NOTEBOOK_NAMESPACE = load_notebook_code(CONFIG.get('notebook_path', 'dots_and_boxes.ipynb'))
    
    return NOTEBOOK_NAMESPACE


def get_game_class():
    """Get the game class from notebook"""
    ns = get_notebook_namespace()
    
    if ns is None:
        return None
    
    # Try to find the game class (could be named differently in your notebook)
    possible_names = ['DotsAndBoxesGame', 'DotsAndBoxes', 'Game', 'GameState']
    
    for name in possible_names:
        if name in ns:
            return ns[name]
    
    # If not found by common names, look for any class with relevant methods
    for name, obj in ns.items():
        if isinstance(obj, type) and hasattr(obj, 'set_edge') and hasattr(obj, 'check_box'):
            print(f"Found game class: {name}")
            return obj
    
    print("ERROR: Could not find game class in notebook")
    return None


def get_ai_class():
    """Get the AI class from notebook"""
    ns = get_notebook_namespace()
    
    if ns is None:
        return None
    
    # Try to find AI class
    possible_names = ['SimpleAI', 'AI', 'MCTSAgent', 'Agent']
    
    for name in possible_names:
        if name in ns:
            return ns[name]
    
    # Look for any class with get_move method
    for name, obj in ns.items():
        if isinstance(obj, type) and hasattr(obj, 'get_move'):
            print(f"Found AI class: {name}")
            return obj
    
    return None


# Game sessions storage
games = {}

@app.route('/')
def index():
    """Serve the main game page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    ns = get_notebook_namespace()
    game_class = get_game_class()
    
    return jsonify({
        'status': 'ok',
        'notebook_loaded': ns is not None,
        'game_class_found': game_class is not None,
        'notebook_path': CONFIG.get('notebook_path')
    })

@app.route('/api/reload', methods=['POST'])
@require_api_key
def reload_notebook():
    """Reload the notebook (useful during development)"""
    global NOTEBOOK_NAMESPACE
    NOTEBOOK_NAMESPACE = None
    
    ns = get_notebook_namespace()
    game_class = get_game_class()
    
    return jsonify({
        'status': 'reloaded',
        'notebook_loaded': ns is not None,
        'game_class_found': game_class is not None
    })

@app.route('/api/new_game', methods=['POST'])
@require_api_key
def new_game():
    """Create a new game"""
    GameClass = get_game_class()
    
    if GameClass is None:
        return jsonify({'error': 'Game class not found in notebook'}), 500
    
    data = request.json or {}
    grid_size = data.get('grid_size', 3)
    game_id = str(random.randint(100000, 999999))
    
    try:
        # Create game instance
        game = GameClass(n=grid_size)
        games[game_id] = game
        
        # Get state (method might vary based on your notebook)
        if hasattr(game, 'get_state'):
            state = game.get_state()
        else:
            # Fallback: construct state manually
            state = {
                'board': game.board_tensor.tolist() if hasattr(game, 'board_tensor') else [],
                'current_player': getattr(game, 'current_player', 1),
                'score_A': len(getattr(game, 'boxes_playerA', [])),
                'score_B': len(getattr(game, 'boxes_playerB', [])),
                'game_over': getattr(game, 'game_over', False),
                'grid_size': grid_size
            }
        
        return jsonify({
            'game_id': game_id,
            'state': state
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error creating game: {str(e)}'}), 500

@app.route('/api/make_move', methods=['POST'])
@require_api_key
def make_move():
    """Make a move in the game"""
    data = request.json
    game_id = data.get('game_id')
    row = data.get('row')
    col = data.get('col')
    edge = data.get('edge')
    
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    game = games[game_id]
    
    try:
        # Call player_turn or equivalent method
        if hasattr(game, 'player_turn'):
            result = game.player_turn(row, col, edge)
        else:
            # Fallback: call methods separately
            game.set_edge(row, col, edge)
            boxes_completed = game.check_box(row, col)
            result = {
                'valid': True,
                'boxes_completed': boxes_completed,
                'game_over': getattr(game, 'game_over', False)
            }
        
        # Get updated state
        if hasattr(game, 'get_state'):
            state = game.get_state()
        else:
            state = {
                'board': game.board_tensor.tolist() if hasattr(game, 'board_tensor') else [],
                'current_player': getattr(game, 'current_player', 1),
                'score_A': len(getattr(game, 'boxes_playerA', [])),
                'score_B': len(getattr(game, 'boxes_playerB', [])),
                'game_over': getattr(game, 'game_over', False)
            }
        
        return jsonify({
            'result': result,
            'state': state
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error making move: {str(e)}'}), 500

@app.route('/api/ai_move', methods=['POST'])
@require_api_key
def ai_move():
    """Get AI move"""
    data = request.json
    game_id = data.get('game_id')
    difficulty = data.get('difficulty', 'medium')
    
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    game = games[game_id]
    AIClass = get_ai_class()
    
    try:
        # Get AI move
        if AIClass and hasattr(AIClass, 'get_move'):
            # Static method or class with get_move
            if hasattr(AIClass.get_move, '__self__'):
                move = AIClass.get_move(game, difficulty)
            else:
                ai = AIClass()
                move = ai.get_move(game, difficulty)
        else:
            # Fallback: random move
            moves = game.get_available_moves() if hasattr(game, 'get_available_moves') else []
            move = random.choice(moves) if moves else None
        
        if move is None:
            return jsonify({'error': 'No moves available'}), 400
        
        r, c, edge = move
        
        # Execute the move
        if hasattr(game, 'player_turn'):
            result = game.player_turn(r, c, edge)
        else:
            game.set_edge(r, c, edge)
            boxes_completed = game.check_box(r, c)
            result = {
                'valid': True,
                'boxes_completed': boxes_completed
            }
        
        # Get state
        if hasattr(game, 'get_state'):
            state = game.get_state()
        else:
            state = {
                'board': game.board_tensor.tolist(),
                'current_player': game.current_player,
                'score_A': len(game.boxes_playerA),
                'score_B': len(game.boxes_playerB),
                'game_over': game.game_over
            }
        
        return jsonify({
            'move': {'row': r, 'col': c, 'edge': edge},
            'result': result,
            'state': state
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error getting AI move: {str(e)}'}), 500

@app.route('/api/get_state/<game_id>', methods=['GET'])
@require_api_key
def get_state(game_id):
    """Get current game state"""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    game = games[game_id]
    
    if hasattr(game, 'get_state'):
        return jsonify(game.get_state())
    else:
        state = {
            'board': game.board_tensor.tolist(),
            'current_player': game.current_player,
            'score_A': len(game.boxes_playerA),
            'score_B': len(game.boxes_playerB),
            'game_over': game.game_over
        }
        return jsonify(state)


if __name__ == '__main__':
    print("=" * 60)
    print("🎮 Dots and Boxes - Dynamic Notebook API Server")
    print("=" * 60)
    
    # Check if notebook exists
    notebook_path = CONFIG.get('notebook_path', 'dots_and_boxes.ipynb')
    if not os.path.exists(notebook_path):
        print(f"\n⚠️  WARNING: Notebook not found at: {notebook_path}")
        print("Please update the 'notebook_path' in api_config.json")
        print("Available .ipynb files:")
        for f in os.listdir('.'):
            if f.endswith('.ipynb'):
                print(f"  - {f}")
        print()
    
    # Try to load notebook
    ns = get_notebook_namespace()
    if ns:
        print("✅ Notebook loaded successfully!")
        game_class = get_game_class()
        if game_class:
            print(f"✅ Game class found: {game_class.__name__}")
        else:
            print("⚠️  Warning: Game class not found")
    else:
        print("❌ Failed to load notebook")
    
    print()
    print(f"Server starting on {CONFIG.get('host')}:{CONFIG.get('port')}")
    print(f"API Key authentication: {'Enabled' if CONFIG.get('api_key') else 'Disabled (dev mode)'}")
    print("=" * 60)
    print()
    
    app.run(
        debug=True,
        host=CONFIG.get('host', '0.0.0.0'),
        port=CONFIG.get('port', 5000)
    )
