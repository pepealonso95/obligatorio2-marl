from base.game import AlternatingGame, AgentID, ActionType
from base.agent import Agent
from math import log, sqrt
import numpy as np
from typing import Callable
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

class MCTSNode:
    def __init__(self, parent: 'MCTSNode', game: AlternatingGame, action: ActionType):
        self.parent = parent
        self.game = game
        self.action = action
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.value = 0
        
        if TORCH_AVAILABLE and DEVICE.type != "cpu":
            self.cum_rewards = torch.zeros(len(game.agents), device=DEVICE, dtype=torch.float32)
            self.use_gpu = True
        else:
            self.cum_rewards = np.zeros(len(game.agents))
            self.use_gpu = False
            
        self.agent = self.game.agent_selection

def ucb(node, C=sqrt(2)) -> float:
    agent_idx = node.game.agent_name_mapping[node.agent]
    if node.use_gpu:
        exploitation = float(node.cum_rewards[agent_idx] / node.visits)
    else:
        exploitation = node.cum_rewards[agent_idx] / node.visits
    
    exploration = C * sqrt(log(node.parent.visits)/node.visits)
    return exploitation + exploration

def uct(node: MCTSNode, agent: AgentID) -> MCTSNode:
    child = max(node.children, key=ucb)
    return child

class MonteCarloTreeSearch(Agent):
    def __init__(self, game: AlternatingGame, agent: AgentID, simulations: int=100, rollouts: int=10, selection: Callable[[MCTSNode, AgentID], MCTSNode]=uct) -> None:
        super().__init__(game=game, agent=agent)
        self.simulations = simulations
        self.rollouts = rollouts
        self.selection = selection
        
        self.use_gpu = TORCH_AVAILABLE and DEVICE.type != "cpu"
        self.device = DEVICE if TORCH_AVAILABLE else None
        
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
        
        if not hasattr(MonteCarloTreeSearch, '_gpu_info_shown'):
            print(f"🔧 MCTS usando: {GPU_TYPE}")
            MonteCarloTreeSearch._gpu_info_shown = True
        
    def action(self) -> ActionType:
        a, _ = self.mcts()
        return a

    def mcts(self) -> tuple[ActionType, float]:

        root = MCTSNode(parent=None, game=self.game, action=None)

        for i in range(self.simulations):

            node = root
            node.game = self.game.clone()

            node = self.select_node(node=node)

            self.expand_node(node)

            rewards = self.rollout(node)

            self.backprop(node, rewards)

        action, value = self.action_selection(root)

        return action, value

    def backprop(self, node, rewards):
        current = node
        while current is not None:
            current.visits += 1
            
            if current.use_gpu:
                if isinstance(rewards, np.ndarray):
                    rewards_tensor = torch.from_numpy(rewards).to(device=DEVICE, dtype=torch.float32)
                    current.cum_rewards += rewards_tensor
                else:
                    current.cum_rewards += rewards
            else:
                if torch.is_tensor(rewards):
                    current.cum_rewards += rewards.cpu().numpy()
                else:
                    current.cum_rewards += rewards
            
            current = current.parent

    def rollout(self, node):
        if self.use_gpu and self.rollouts > 3:
            return self._rollout_gpu_optimized(node)
        else:
            return self._rollout_standard(node)
    
    def _rollout_standard(self, node):
        rewards = np.zeros(len(self.game.agents))
        
        for _ in range(self.rollouts):
            game_copy = node.game.clone()
            while not game_copy.game_over():
                available_actions = game_copy.available_actions()
                if available_actions:
                    action = np.random.choice(available_actions)
                    game_copy.step(action)
            rollout_rewards = np.array([game_copy.reward(agent) for agent in game_copy.agents])
            rewards += rollout_rewards
        
        return rewards / self.rollouts
    
    def _rollout_gpu_optimized(self, node):
        try:
            rewards_accumulator = torch.zeros(len(self.game.agents), device=self.device, dtype=torch.float32)
            
            for _ in range(self.rollouts):
                game_copy = node.game.clone()
                step_count = 0
                
                while not game_copy.game_over() and step_count < 100:
                    available_actions = game_copy.available_actions()
                    if available_actions:
                        if len(available_actions) > 1:
                            action_idx = torch.randint(0, len(available_actions), (1,), device=self.device).item()
                            action = available_actions[action_idx]
                        else:
                            action = available_actions[0]
                        game_copy.step(action)
                        step_count += 1
                
                rollout_rewards = torch.tensor([game_copy.reward(agent) for agent in game_copy.agents], 
                                             device=self.device, dtype=torch.float32)
                rewards_accumulator += rollout_rewards
            
            return rewards_accumulator / self.rollouts
            
        except Exception:
            return self._rollout_standard(node)

    def select_node(self, node: MCTSNode) -> MCTSNode:
        curr_node = node
        while curr_node.children:
            if curr_node.explored_children < len(curr_node.children):
                curr_node = curr_node.children[curr_node.explored_children]
                curr_node.parent.explored_children += 1
                break
            else:
                curr_node = self.selection(curr_node, curr_node.agent)
        return curr_node

    def expand_node(self, node) -> None:
        if not node.game.game_over():
            available_actions = node.game.available_actions()
            if available_actions:
                for action in available_actions:
                    child_game = node.game.clone()
                    child_game.step(action)
                    child_node = MCTSNode(parent=node, game=child_game, action=action)
                    node.children.append(child_node)

    def action_selection(self, node: MCTSNode):
        action = None
        value = 0
        
        if node.children:
            agent_idx = node.game.agent_name_mapping[self.agent]
            
            if self.use_gpu and len(node.children) > 2:
                try:
                    child_values = []
                    for child in node.children:
                        if child.use_gpu:
                            child_value = float(child.cum_rewards[agent_idx] / max(child.visits, 1))
                        else:
                            child_value = child.cum_rewards[agent_idx] / max(child.visits, 1)
                        child_values.append(child_value)
                    
                    values_tensor = torch.tensor(child_values, device=self.device, dtype=torch.float32)
                    best_idx = torch.argmax(values_tensor).item()
                    
                    best_child = node.children[best_idx]
                    action = best_child.action
                    value = child_values[best_idx]
                    
                except Exception:
                    best_child = max(node.children, key=lambda child: child.cum_rewards[agent_idx] / max(child.visits, 1))
                    action = best_child.action
                    value = best_child.cum_rewards[agent_idx] / max(best_child.visits, 1)
            else:
                best_child = max(node.children, key=lambda child: child.cum_rewards[agent_idx] / max(child.visits, 1))
                action = best_child.action
                
                if best_child.use_gpu:
                    value = float(best_child.cum_rewards[agent_idx] / max(best_child.visits, 1))
                else:
                    value = best_child.cum_rewards[agent_idx] / max(best_child.visits, 1)
        
        return action, value

    def get_performance_info(self):
        return {
            'device': str(self.device) if self.device else 'CPU',
            'gpu_enabled': self.use_gpu,
            'torch_available': TORCH_AVAILABLE,
            'simulations': self.simulations,
            'rollouts': self.rollouts
        }    