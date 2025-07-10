from base.agent import Agent, AgentID
from base.game import AlternatingGame
import numpy as np
import sys
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        GPU_TYPE = "MPS (Apple Silicon)"
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        GPU_TYPE = "CUDA"
    else:
        DEVICE = torch.device("cpu")
        GPU_TYPE = "CPU"
        
    warnings.filterwarnings('ignore', category=UserWarning)
    
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    GPU_TYPE = "CPU (PyTorch not available)"

class MiniMax(Agent):

    def __init__(self, game: AlternatingGame, agent: AgentID, seed=None, depth: int=sys.maxsize) -> None:
        super().__init__(game, agent)

        if depth < 0:
            raise ValueError("Depth must be a non-negative integer.")

        self.depth = depth
        self.seed = seed
        
        self.use_gpu = TORCH_AVAILABLE and DEVICE.type != "cpu"
        self.device = DEVICE if TORCH_AVAILABLE else None
        
        np.random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed if seed is not None else 42)
        
        self._eval_cache = {} if self.use_gpu else None
        
        if not hasattr(MiniMax, '_gpu_info_shown'):
            print(f"🔧 MiniMax usando: {GPU_TYPE}")
            MiniMax._gpu_info_shown = True
    
    def action(self):
        act, _ = self.minimax(self.game, self.depth)
        return act

    def minimax(self, game: AlternatingGame, depth: int):

        agent = game.agent_selection
        chosen_action = None

        if game.terminated():             
            return None, game.reward(self.agent)

        if depth == 0:
            return None, self.eval(game)
        
        actions = game.available_actions()
        np.random.shuffle(actions)
        action_nodes = []
        for action in actions:
            child = game.clone()
            child.step(action)
            action_nodes.append((action, child))

        if self.use_gpu and len(action_nodes) > 2:
            return self._minimax_gpu_optimized(action_nodes, depth, agent)
        else:
            return self._minimax_standard(action_nodes, depth, agent)

    def _minimax_standard(self, action_nodes, depth, agent):
        chosen_action = None
        
        if agent != self.agent:
            value = float('inf')
            for action, child in action_nodes:
                _, minimax_value = self.minimax(child, depth-1)
                if minimax_value < value:
                    value = minimax_value
                    chosen_action = action
        else:
            value = float('-inf')
            for action, child in action_nodes:
                _, minimax_value = self.minimax(child, depth-1)
                if minimax_value > value:
                    value = minimax_value
                    chosen_action = action

        return chosen_action, value

    def _minimax_gpu_optimized(self, action_nodes, depth, agent):
        try:
            values = []
            actions = []
            
            for action, child in action_nodes:
                _, minimax_value = self.minimax(child, depth-1)
                values.append(float(minimax_value))
                actions.append(action)
            
            values_tensor = torch.tensor(values, device=self.device, dtype=torch.float32)
            
            if agent != self.agent:
                min_idx = torch.argmin(values_tensor).item()
                chosen_action = actions[min_idx]
                value = values[min_idx]
            else:
                max_idx = torch.argmax(values_tensor).item()
                chosen_action = actions[max_idx]
                value = values[max_idx]
            
            return chosen_action, value
            
        except Exception:
            return self._minimax_standard(action_nodes, depth, agent)

    def eval(self, game: AlternatingGame):
        if self._eval_cache is not None:
            game_state = str(game.state) if hasattr(game, 'state') else None
            if game_state and game_state in self._eval_cache:
                return self._eval_cache[game_state]
        
        if hasattr(game, 'eval'):
            result = game.eval(self.agent)
        else:
            if game.terminated():
                result = game.reward(self.agent)
            else:
                result = 0.0
        
        if self._eval_cache is not None and len(self._eval_cache) < 1000:
            game_state = str(game.state) if hasattr(game, 'state') else None
            if game_state:
                self._eval_cache[game_state] = result
        
        return result

    def get_performance_info(self):
        info = {
            'device': str(self.device) if self.device else 'CPU',
            'gpu_enabled': self.use_gpu,
            'torch_available': TORCH_AVAILABLE
        }
        if self._eval_cache is not None:
            info['cache_size'] = len(self._eval_cache)
        return info

    def clear_cache(self):
        if self._eval_cache is not None:
            self._eval_cache.clear()