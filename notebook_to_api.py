from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from functools import wraps
import json
import os
import sys
import nbformat
import random
import numpy as np

app = Flask(__name__)
CORS(app)

# Load API configuration
def load_config():
    """Load API configuration from file or environment"""
    config = {
        'api_key': os.environ.get('API_KEY'),
        'notebook_path': os.environ.get('NOTEBOOK_PATH', 'Dots_and_Boxes_Model.ipynb'), 
        'port': int(os.environ.get('PORT', 5000)),
        'host': os.environ.get('HOST', '0.0.0.0')
    }
    
    if os.path.exists('api_config.json'):
        try:
            with open('api_config.json', 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            print(f"Warning: Could not load api_config.json: {e}")
    
    return config

CONFIG = load_config()

# API Key Authentication
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if os.environ.get('FLASK_ENV') == 'development':
            return f(*args, **kwargs)
            
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        if CONFIG.get('api_key') and api_key != CONFIG['api_key']:
            return jsonify({'error': 'Invalid API key'}), 403
            
        return f(*args, **kwargs)
    return decorated_function


def should_skip_cell(code):
    """Skip setup and training cells"""
    code = code.strip()
    
    if code.startswith('%') or code.startswith('!'):
        return True
    
    if 'setup_api(' in code and not code.startswith('#'):
        return True
    
    if 'generate_api_key()' in code:
        return True
    
    if 'update_index_html_api_key' in code:
        return True
    
    # Skip training - it's too slow
    if 'trainer.train()' in code or 'Trainer(' in code:
        return True
    
    return False


def load_notebook_code(notebook_path):
    """Load and execute code from Jupyter notebook"""
    print(f"Loading notebook: {notebook_path}")
    
    if not os.path.exists(notebook_path):
        print(f"ERROR: Notebook not found at {notebook_path}")
        return None
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        notebook_globals = {}
        cells_executed = 0
        cells_skipped = 0
        
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                code = cell.source
                
                if should_skip_cell(code):
                    cells_skipped += 1
                    continue
                
                try:
                    exec(code, notebook_globals)
                    cells_executed += 1
                except Exception as e:
                    print(f"Warning: Error executing cell {i}: {e}")
                    continue
        
        print(f"Notebook loaded: {cells_executed} cells executed, {cells_skipped} skipped")
        return notebook_globals
        
    except Exception as e:
        print(f"ERROR loading notebook: {e}")
        import traceback
        traceback.print_exc()
        return None


# Global notebook namespace
NOTEBOOK_NAMESPACE = None

def get_notebook_namespace():
    """Get or reload notebook namespace"""
    global NOTEBOOK_NAMESPACE
    
    if NOTEBOOK_NAMESPACE is None:
        NOTEBOOK_NAMESPACE = load_notebook_code(CONFIG.get('notebook_path'))
    
    return NOTEBOOK_NAMESPACE


def get_class_from_notebook(class_name):
    """Get a class from the loaded notebook"""
    ns = get_notebook_namespace()
    if ns and class_name in ns:
        return ns[class_name]
    return None


# Game wrapper with state tracking
class GameWrapper:
    """Wrapper around dots_and_boxes with AI integration"""
    
    def __init__(self, game, grid_size, mcts=None, model=None, device=None, num_simulations=100):
        self.game = game
        self.grid_size = grid_size
        self.current_player = 1
        self.move_count = 0
        self.mcts_class = mcts
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.using_model = model is not None
        
    def check_edge_available(self, r, c, edge):
        """Check if an edge is available"""
        return self.game.check_edge(r, c, edge)
    
    def make_move(self, r, c, edge):
        """Make a move"""
        if not self.check_edge_available(r, c, edge):
            return {
                'valid': False,
                'error': 'Edge already drawn',
                'boxes_completed': 0
            }
        
        self.move_count += 1
        boxes_completed = self.game.player_turn(r, c, edge, self.current_player, self.move_count)
        
        if boxes_completed == 0:
            self.current_player = 3 - self.current_player
        
        return {
            'valid': True,
            'boxes_completed': int(boxes_completed),
            'game_over': bool(self.is_game_over()),
            'using_trained_model': self.using_model
        }
    
    def get_ai_move(self):
        """Get AI move using the trained model"""
        if self.mcts_class is None or self.model is None:
            print(f"No model available for {self.grid_size}x{self.grid_size}, using random move")
            return self._get_random_move()
        
        try:
            # Create MCTS instance with your model
            mcts = self.mcts_class(
                model=self.model,
                device=self.device,
                num_simulations=self.num_simulations
            )
            
            # Get move from MCTS
            action_probs = mcts.search(self.game, self.current_player)
            action = np.argmax(action_probs)
            r, c, edge_type = mcts.decode_action(action, self.grid_size)
            
            # Log confidence
            confidence = action_probs[action]
            print(f"AI move: ({r},{c}) {edge_type} with confidence {confidence:.3f}")
            
            return (r, c, edge_type)
            
        except Exception as e:
            print(f"Error in MCTS: {e}")
            import traceback
            traceback.print_exc()
            return self._get_random_move()
    
    def _get_random_move(self):
        """Fallback: random move"""
        available = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                for edge in ['top', 'bottom', 'left', 'right']:
                    if self.check_edge_available(r, c, edge):
                        available.append((r, c, edge))
        return random.choice(available) if available else None
    
    def is_game_over(self):
        """Check if game is over"""
        total_boxes = self.grid_size * self.grid_size
        claimed = len(self.game.boxes_playerA) + len(self.game.boxes_playerB)
        return claimed >= total_boxes
    
    def get_state(self):
        """Get current game state"""
        return {
            'board': self.game.board_tensor.tolist(),
            'current_player': int(self.current_player),
            'score_A': int(len(self.game.boxes_playerA)),
            'score_B': int(len(self.game.boxes_playerB)),
            'game_over': bool(self.is_game_over()),
            'grid_size': int(self.grid_size),
            'move_count': int(self.move_count),
            'using_trained_model': self.using_model
        }


# Global model storage - support multiple board sizes
DEVICE = None
MODEL_CACHE = {}  # {board_size: model}
MODEL_INFO_CACHE = {}  # {board_size: info_dict}
games = {}

def load_trained_model_for_size(board_size):
    """
    Load a trained model for a specific board size.
    Supports multiple naming conventions:
    - best_model_NxN.pth (e.g., best_model_2x2.pth, best_model_3x3.pth)
    - best_model.pth (default, tries to detect size)
    - model_NxN.pth
    - model.pth
    """
    global DEVICE, MODEL_CACHE, MODEL_INFO_CACHE
    
    # Check cache first
    if board_size in MODEL_CACHE:
        return MODEL_CACHE[board_size], MODEL_INFO_CACHE[board_size]
    
    try:
        import torch
        if DEVICE is None:
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {DEVICE}")
        
        # Get model class from notebook
        Connect2Model = get_class_from_notebook('Connect2Model')
        if not Connect2Model:
            print("ERROR: Connect2Model class not found in notebook")
            return None, None
        
        # Try different file naming patterns
        model_files = [
            f'best_model_{board_size}x{board_size}.pth',
            f'model_{board_size}x{board_size}.pth',
            'best_model.pth',
            'model.pth'
        ]
        
        for model_file in model_files:
            if not os.path.exists(model_file):
                continue
            
            print(f"\nAttempting to load: {model_file} for {board_size}x{board_size} board")
            print(f"File size: {os.path.getsize(model_file):,} bytes")
            
            try:
                checkpoint = torch.load(model_file, map_location=DEVICE)
                
                # Extract state dict
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                    
                    metadata = checkpoint if isinstance(checkpoint, dict) else {}
                else:
                    state_dict = checkpoint
                    metadata = {}
                
                # Detect board size from model architecture
                detected_size = None
                if 'policy_fc.weight' in state_dict:
                    policy_out_features = state_dict['policy_fc.weight'].shape[0]
                    from math import sqrt
                    n = int((-2 + sqrt(4 + 8 * policy_out_features)) / 4)
                    if 2 * n * (n + 1) == policy_out_features:
                        detected_size = n
                
                # If this model doesn't match requested size, skip
                if detected_size and detected_size != board_size:
                    print(f"  Skipping: detected size {detected_size}x{detected_size}, need {board_size}x{board_size}")
                    continue
                
                # Check if model appears trained (not just random initialization)
                param_values = []
                for key, param in state_dict.items():
                    if 'weight' in key:
                        param_values.extend(param.flatten().cpu().numpy()[:100].tolist())
                
                if len(param_values) > 0:
                    mean_abs = np.mean(np.abs(param_values))
                    std = np.std(param_values)
                    if mean_abs < 0.001 or std < 0.001:
                        print(f"  Skipping: appears to be untrained (near-zero weights)")
                        continue
                
                # Initialize model with correct size
                print(f"  Creating model for {board_size}x{board_size}")
                model = Connect2Model(board_size=(board_size, board_size), device=DEVICE)
                
                # Load state dict
                model.load_state_dict(state_dict)
                model.eval()
                
                # Test forward pass
                with torch.no_grad():
                    test_input = torch.zeros(1, 6, board_size, board_size).to(DEVICE)
                    policy, value = model(test_input)
                    expected_actions = 2 * board_size * (board_size + 1)
                    
                    if policy.shape[1] != expected_actions:
                        print(f"  Skipping: policy shape mismatch")
                        continue
                
                # Build model info
                model_info = {
                    'loaded': True,
                    'file': model_file,
                    'board_size': board_size,
                    'parameters': sum(p.numel() for p in state_dict.values()),
                    'win_rate': metadata.get('win_rate'),
                    'training_step': metadata.get('training_step')
                }
                
                # Cache and return
                MODEL_CACHE[board_size] = model
                MODEL_INFO_CACHE[board_size] = model_info
                
                print(f"  SUCCESS: Loaded {model_file} for {board_size}x{board_size}")
                print(f"  Parameters: {model_info['parameters']:,}")
                if model_info['win_rate']:
                    print(f"  Win rate: {model_info['win_rate']:.2%}")
                
                return model, model_info
                
            except Exception as e:
                print(f"  Error loading {model_file}: {e}")
                continue
        
        # No model found for this size
        print(f"No trained model found for {board_size}x{board_size} board")
        return None, None
        
    except Exception as e:
        print(f"Error in load_trained_model_for_size: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def get_available_board_sizes():
    """Get list of board sizes that have trained models available"""
    sizes = []
    
    # Check for size-specific models
    for size in [2, 3, 4, 5]:
        for pattern in [f'best_model_{size}x{size}.pth', f'model_{size}x{size}.pth']:
            if os.path.exists(pattern):
                sizes.append(size)
                break
    
    # Check default models
    for default_file in ['best_model.pth', 'model.pth']:
        if os.path.exists(default_file):
            # Try to detect size
            try:
                import torch
                checkpoint = torch.load(default_file, map_location='cpu')
                if isinstance(checkpoint, dict):
                    state_dict = checkpoint.get('model_state_dict', checkpoint)
                else:
                    state_dict = checkpoint
                
                if 'policy_fc.weight' in state_dict:
                    policy_out = state_dict['policy_fc.weight'].shape[0]
                    from math import sqrt
                    n = int((-2 + sqrt(4 + 8 * policy_out)) / 4)
                    if 2 * n * (n + 1) == policy_out and n not in sizes:
                        sizes.append(n)
            except:
                pass
    
    return sorted(sizes) if sizes else []


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    available_sizes = get_available_board_sizes()
    return jsonify({
        'status': 'ok',
        'notebook_loaded': NOTEBOOK_NAMESPACE is not None,
        'available_board_sizes': available_sizes,
        'cached_models': list(MODEL_CACHE.keys()),
        'model_info': MODEL_INFO_CACHE
    })

@app.route('/api/new_game', methods=['POST'])
@require_api_key
def new_game():
    GameClass = get_class_from_notebook('dots_and_boxes')
    if not GameClass:
        return jsonify({'error': 'Game class not found'}), 500
    
    data = request.json or {}
    grid_size = data.get('grid_size', 2)
    num_simulations = data.get('num_simulations', 100)
    starting_player = data.get('starting_player', 2)  # Default to AI starting (player 2)
    game_id = str(random.randint(100000, 999999))
    
    try:
        # Load model for this board size (or None if not available)
        model, model_info = load_trained_model_for_size(grid_size)
        
        raw_game = GameClass(n=grid_size)
        MCTSClass = get_class_from_notebook('MCTS')
        
        game = GameWrapper(
            raw_game, 
            grid_size,
            mcts=MCTSClass,
            model=model,
            device=DEVICE,
            num_simulations=num_simulations
        )
        
        # Set starting player
        game.current_player = starting_player
        
        games[game_id] = game
        
        model_status = "YES" if model else "NO (random moves)"
        print(f"New game created: {game_id} (grid={grid_size}, sims={num_simulations}, model={model_status}, starting_player={starting_player})")
        
        return jsonify({
            'game_id': game_id,
            'state': game.get_state(),
            'model_loaded': model is not None,
            'model_info': model_info,
            'num_simulations': num_simulations
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/make_move', methods=['POST'])
@require_api_key
def make_move():
    data = request.json
    game_id = data.get('game_id')
    row = data.get('row')
    col = data.get('col')
    edge = data.get('edge')
    
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    game = games[game_id]
    
    try:
        result = game.make_move(row, col, edge)
        state = game.get_state()
        
        print(f"Player move: ({row},{col}) {edge} -> {result['boxes_completed']} boxes")
        
        return jsonify({
            'result': result,
            'state': state
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai_move', methods=['POST'])
@require_api_key
def ai_move():
    data = request.json
    game_id = data.get('game_id')
    
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    game = games[game_id]
    
    try:
        move = game.get_ai_move()
        
        if move is None:
            return jsonify({'error': 'No moves available'}), 400
        
        r, c, edge = move
        result = game.make_move(r, c, edge)
        state = game.get_state()
        
        return jsonify({
            'move': {'row': int(r), 'col': int(c), 'edge': str(edge)},
            'result': result,
            'state': state
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_state/<game_id>', methods=['GET'])
@require_api_key
def get_state(game_id):
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    return jsonify(games[game_id].get_state())


if __name__ == '__main__':
    print("=" * 80)
    print("DOTS AND BOXES - Multi-Board Size Server")
    print("=" * 80)
    
    # Load notebook
    ns = get_notebook_namespace()
    if ns:
        print("Notebook loaded successfully")
        
        if get_class_from_notebook('dots_and_boxes'):
            print("Game class found")
        
        if get_class_from_notebook('MCTS'):
            print("MCTS class found")
        
        # Check available models
        print("\n" + "=" * 80)
        print(" Checking Available Models")
        print("=" * 80)
        
        available_sizes = get_available_board_sizes()
        if available_sizes:
            print(f"\nFound models for board sizes: {available_sizes}")
            print("\nPre-loading models...")
            for size in available_sizes:
                model, info = load_trained_model_for_size(size)
                if model:
                    print(f"  {size}x{size}: Loaded successfully")
        else:
            print("\nNo trained models found")
            print("AI will play randomly for all board sizes")
            print("\nTo add trained models:")
            print("   1. Train model: trainer.train(max_training_cycles=500)")
            print("   2. Save as: best_model_NxN.pth (e.g., best_model_2x2.pth)")
            print("   3. Place in this directory and restart server")
    else:
        print("Failed to load notebook")
    
    print("\n" + "=" * 80)
    print(f" Server starting at http://{CONFIG.get('host')}:{CONFIG.get('port')}")
    print("=" * 80)
    print()
    
    app.run(
        debug=True,
        host=CONFIG.get('host', '0.0.0.0'),
        port=CONFIG.get('port', 5000)
    )
