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
            print("No model available, using random move")
            return self._get_random_move()
        
        try:
            # Create MCTS instance with your model
            mcts = self.mcts_class(
                model=self.model,
                device=self.device,
                num_simulations=self.num_simulations
            )
            
            # Get move from MCTS (exactly like in your notebook)
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


# Game sessions
games = {}

# Load model on startup
TRAINED_MODEL = None
DEVICE = None
MODEL_INFO = {
    'loaded': False,
    'file': None,
    'win_rate': None,
    'training_step': None,
    'parameters': 0
}

def load_trained_model():
    """Load the trained model with detailed diagnostics"""
    global TRAINED_MODEL, DEVICE, MODEL_INFO
    
    try:
        import torch
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {DEVICE}")
        
        Connect2Model = get_class_from_notebook('Connect2Model')
        if not Connect2Model:
            print("Connect2Model class not found in notebook")
            return None
        
        # Try to load saved model
        model_files = ['best_model.pth', 'model.pth', 'checkpoint.pth']
        for model_file in model_files:
            if os.path.exists(model_file):
                print(f"Found model file: {model_file} ({os.path.getsize(model_file)} bytes)")
                
                checkpoint = torch.load(model_file, map_location=DEVICE)
                
                # Extract model info
                MODEL_INFO['file'] = model_file
                if isinstance(checkpoint, dict):
                    print(f"Checkpoint type: dictionary")
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                        print(f"Found model_state_dict")
                    else:
                        state_dict = checkpoint
                        print(f"Using checkpoint as state_dict directly")
                    
                    if 'win_rate' in checkpoint:
                        MODEL_INFO['win_rate'] = checkpoint['win_rate']
                        print(f"Model win rate: {checkpoint['win_rate']:.2%}")
                    if 'training_step' in checkpoint:
                        MODEL_INFO['training_step'] = checkpoint['training_step']
                        print(f"Training step: {checkpoint['training_step']}")
                else:
                    state_dict = checkpoint
                
                # Count parameters
                MODEL_INFO['parameters'] = sum(p.numel() for p in state_dict.values())
                print(f"Total parameters: {MODEL_INFO['parameters']:,}")
                
                # Initialize model
                model = Connect2Model(board_size=(2, 2), device=DEVICE)
                model.load_state_dict(state_dict)
                model.eval()
                
                # Test forward pass
                with torch.no_grad():
                    test_input = torch.zeros(1, 6, 2, 2).to(DEVICE)
                    policy, value = model(test_input)
                    print(f"Forward pass successful: policy {policy.shape}, value {value.shape}")
                
                TRAINED_MODEL = model
                MODEL_INFO['loaded'] = True
                print(f"Model loaded and verified from {model_file}")
                return model
        
        print("No model file found")
        print("To use a trained model:")
        print("   1. Train in your notebook: trainer.train(max_training_cycles=500)")
        print("   2. Download 'best_model.pth' from Colab")
        print("   3. Place it in this directory")
        print("   4. Restart the server")
        return None
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': TRAINED_MODEL is not None,
        'notebook_loaded': NOTEBOOK_NAMESPACE is not None,
        'model_info': MODEL_INFO
    })

@app.route('/api/new_game', methods=['POST'])
@require_api_key
def new_game():
    GameClass = get_class_from_notebook('dots_and_boxes')
    if not GameClass:
        return jsonify({'error': 'Game class not found'}), 500
    
    data = request.json or {}
    grid_size = data.get('grid_size', 2)
    num_simulations = data.get('num_simulations', 100)  # Increased default
    game_id = str(random.randint(100000, 999999))
    
    try:
        raw_game = GameClass(n=grid_size)
        MCTSClass = get_class_from_notebook('MCTS')
        
        game = GameWrapper(
            raw_game, 
            grid_size,
            mcts=MCTSClass,
            model=TRAINED_MODEL,
            device=DEVICE,
            num_simulations=num_simulations
        )
        games[game_id] = game
        
        print(f"New game created: {game_id} (grid={grid_size}, sims={num_simulations}, model={'YES' if TRAINED_MODEL else 'NO'})")
        
        return jsonify({
            'game_id': game_id,
            'state': game.get_state(),
            'model_loaded': TRAINED_MODEL is not None,
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
    print("DOTS AND BOXES - Enhanced Server with Diagnostics")
    print("=" * 80)
    
    # Load notebook
    ns = get_notebook_namespace()
    if ns:
        print("Notebook loaded successfully")
        
        if get_class_from_notebook('dots_and_boxes'):
            print("Game class found")
        
        if get_class_from_notebook('MCTS'):
            print("MCTS class found")
        
        # Load model
        print("\n" + "=" * 80)
        print(" Loading AI Model")
        print("=" * 80)
        model = load_trained_model()
        
        if model:
            print("\nUSING TRAINED MODEL")
            print(f"   • Model file: {MODEL_INFO['file']}")
            print(f"   • Parameters: {MODEL_INFO['parameters']:,}")
            if MODEL_INFO['win_rate']:
                print(f"   • Win rate: {MODEL_INFO['win_rate']:.2%}")
            if MODEL_INFO['training_step']:
                print(f"   • Training step: {MODEL_INFO['training_step']}")
            print(f"   • Default MCTS simulations: 100")
        else:
            print("\nNO MODEL LOADED - AI WILL PLAY RANDOMLY")
            print("\nTo train a model:")
            print("   1. Open your Colab notebook")
            print("   2. Run: trainer.train(max_training_cycles=500)")
            print("   3. Download 'best_model.pth'")
            print("   4. Place in this directory and restart server")
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
