import numpy as np
import pickle
import os
import time
from numpy import ndarray
from base.game import AlternatingGame, AgentID, ObsType
from base.agent import Agent

class Node():

    def __init__(self, game: AlternatingGame, obs: ObsType) -> None:
        self.game = game
        self.agent = game.agent_selection
        self.obs = obs
        self.num_actions = self.game.num_actions(self.agent)
        
        self.cum_regrets = np.zeros(self.num_actions)
        self.curr_policy = np.full(self.num_actions, 1/self.num_actions)
        self.sum_policy = self.curr_policy.copy()
        self.learned_policy = self.curr_policy.copy()
        
        self.niter = 1

    def regret_matching(self):
        positive_regrets = np.maximum(self.cum_regrets, 0)
        regret_sum = np.sum(positive_regrets)
        
        if regret_sum > 0:
            self.curr_policy = positive_regrets / regret_sum
        else:
            self.curr_policy = np.full(self.num_actions, 1/self.num_actions)
        
        self.sum_policy += self.curr_policy
        self.learned_policy = self.sum_policy / self.niter
    
    def update(self, utility, node_utility, probability) -> None:
        for action in range(self.num_actions):
            self.cum_regrets[action] += (utility[action] - node_utility) * probability
        
        self.niter += 1
        self.regret_matching()

    def policy(self):
        return self.learned_policy

class CounterFactualRegret(Agent):

    def __init__(self, game: AlternatingGame, agent: AgentID, max_depth=50, use_gpu=False) -> None:
        super().__init__(game, agent)
        self.node_dict: dict[ObsType, Node] = {}
        self.max_depth = max_depth
        self.use_gpu = use_gpu
        self.device = 'cpu'
        
        if not hasattr(self.game, 'rewards') or self.game.rewards is None:
            self.game.reset()

    def action(self):
        try:
            obs = self.game.observe(self.agent)
            if obs in self.node_dict:
                node = self.node_dict[obs]
                policy = node.policy()
                available_actions = self.game.available_actions()
                if len(available_actions) == 0:
                    return 0
                
                valid_policy = np.zeros(len(available_actions))
                for i, action in enumerate(available_actions):
                    if action < len(policy):
                        valid_policy[i] = policy[action]
                
                if np.sum(valid_policy) > 0:
                    valid_policy = valid_policy / np.sum(valid_policy)
                else:
                    valid_policy = np.ones(len(available_actions)) / len(available_actions)
                
                chosen_idx = np.random.choice(len(available_actions), p=valid_policy)
                return available_actions[chosen_idx]
            else:
                return np.random.choice(self.game.available_actions())
        except:
            return np.random.choice(self.game.available_actions())
    
    def train(self, niter=1000, timeout_seconds=None):
        start_time = time.time()
        
        for i in range(niter):
            if timeout_seconds is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout_seconds:
                    print(f"Timeout: {timeout_seconds}s, iterations: {i}, nodes: {len(self.node_dict)}, time: {elapsed:.1f}s")
                    return
            
            try:
                _ = self.cfr()
                
                if len(self.node_dict) > 5000:
                    elapsed = time.time() - start_time
                    print(f"STOP: nodes: {len(self.node_dict)}, iteration: {i}, time: {elapsed:.1f}s")
                    break
                
                if i % 100 == 0 and i > 0:  
                    elapsed = time.time() - start_time
                    print(f"Iteration: {i}/{niter}, nodes: {len(self.node_dict)}, time: {elapsed:.1f}s")
                    
            except RecursionError:
                elapsed = time.time() - start_time
                print(f"Recursion limit: iteration {i}, time: {elapsed:.1f}s")
                break
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"Error: iteration {i}, time: {elapsed:.1f}s, {e}")
                break
        
        total_time = time.time() - start_time
        print(f"Training: {niter} iterations, {total_time:.1f}s, {len(self.node_dict)} nodes")

    def cfr(self):
        game = self.game.clone()
        utility: dict[AgentID, float] = dict()
        for agent in self.game.agents:
            game.reset()

            if not hasattr(game, 'rewards') or game.rewards is None:
                game.rewards = {a: 0 for a in game.agents}
            probability = np.ones(game.num_agents)
            try:
                utility[agent] = self.cfr_rec(game=game, agent=agent, probability=probability, depth=0)
            except RecursionError:
                utility[agent] = 0.0 
        return utility 

    # uso la recursion porque es más fácil de entender y manejar en este caso
    # pero se puede optimizar con un stack si es necesario para evitar RecursionError
    def cfr_rec(self, game: AlternatingGame, agent: AgentID, probability: ndarray, depth=0):
        
        if depth >= self.max_depth:
            return self._safe_get_reward(game, agent)
            
        if game.game_over():
            return self._safe_get_reward(game, agent)
        
        try:
            obs = game.observe(game.agent_selection)
        except:
            return 0.0 
            
        if obs not in self.node_dict:
            self.node_dict[obs] = Node(game, obs)
        
        node = self.node_dict[obs]
        available_actions = game.available_actions()
        
        if not available_actions:
            return self._safe_get_reward(game, agent)
        
        if game.agent_selection == agent:

            action_utilities = np.zeros(node.num_actions)
            
            for action in available_actions:
                if action < len(action_utilities): 
                    try:
                        game_copy = game.clone()
                        if not hasattr(game_copy, 'rewards') or game_copy.rewards is None:
                            game_copy.rewards = {a: 0 for a in game_copy.agents}
                        game_copy.step(action)
                        utility_value = self.cfr_rec(game_copy, agent, probability, depth + 1)
                        action_utilities[action] = utility_value
                    except:
                        action_utilities[action] = 0.0
            
            available_policy = self._get_available_policy(node, available_actions)
            
            node_utility = 0.0
            for i, action in enumerate(available_actions):
                if action < len(action_utilities):
                    node_utility += available_policy[i] * action_utilities[action]
            
            try:
                prob_agent = probability[game.agent_name_mapping[agent]]
                
                utility_for_update = np.zeros(node.num_actions)
                for action in available_actions:
                    if action < len(utility_for_update):
                        utility_for_update[action] = action_utilities[action]
                
                node.update(utility_for_update, node_utility, prob_agent)
            except:
                pass
            
            return node_utility
        else:
            available_policy = self._get_available_policy(node, available_actions)
            expected_utility = 0.0
            
            for i, action in enumerate(available_actions):
                try:
                    game_copy = game.clone()
                    if not hasattr(game_copy, 'rewards') or game_copy.rewards is None:
                        game_copy.rewards = {a: 0 for a in game_copy.agents}
                    game_copy.step(action)
                    new_prob = probability.copy()
                    other_agent_idx = game.agent_name_mapping[game.agent_selection]
                    new_prob[other_agent_idx] *= available_policy[i]
                    expected_utility += available_policy[i] * self.cfr_rec(game_copy, agent, new_prob, depth + 1)
                except:
                    continue
            
            return expected_utility
    
    def _safe_get_reward(self, game: AlternatingGame, agent: AgentID):
        try:
            if hasattr(game, 'rewards') and game.rewards is not None:
                return game.reward(agent)
            else:
                return 0.0
        except (AttributeError, KeyError):
            return 0.0
    
    def _get_available_policy(self, node, available_actions):
        if not available_actions:
            return np.array([])
            
        available_policy = []
        for action in available_actions:
            if action < len(node.curr_policy):
                available_policy.append(node.curr_policy[action])
            else:
                available_policy.append(1.0 / len(available_actions))
        
        available_policy = np.array(available_policy)
        
        policy_sum = np.sum(available_policy)
        if policy_sum > 0:
            available_policy = available_policy / policy_sum
        else:
            available_policy = np.ones(len(available_actions)) / len(available_actions)
            
        return available_policy
    
    def save_agent(self, filepath):
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_data = {
            'agent_id': self.agent,
            'max_depth': self.max_depth,
            'nodes': {}
        }
        
        for obs, node in self.node_dict.items():
            obs_key = str(obs)
            
            save_data['nodes'][obs_key] = {
                'cum_regrets': node.cum_regrets.tolist(),
                'sum_policy': node.sum_policy.tolist(),
                'learned_policy': node.learned_policy.tolist(),
                'niter': node.niter,
                'num_actions': node.num_actions
            }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Saved: {filepath}, nodes: {len(self.node_dict)}")
    
    def load_agent(self, filepath, game):
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            self.max_depth = save_data.get('max_depth', 50)
            
            self.node_dict = {}
            
            for obs_key, node_data in save_data['nodes'].items():
                try:
                    if obs_key.startswith('(') and obs_key.endswith(')'):
                        temp_obs = eval(obs_key)
                    elif obs_key.startswith('[') and obs_key.endswith(']'):
                        temp_obs = tuple(eval(obs_key))
                    else:
                        temp_obs = obs_key
                except:
                    temp_obs = obs_key
                
                dummy_game = game.clone()
                node = Node(dummy_game, temp_obs)
                
                node.cum_regrets = np.array(node_data['cum_regrets'])
                node.sum_policy = np.array(node_data['sum_policy'])
                node.learned_policy = np.array(node_data['learned_policy'])
                node.niter = node_data['niter']
                node.num_actions = node_data['num_actions']
                
                node.regret_matching()
                
                self.node_dict[temp_obs] = node
            
            print(f"Loaded: {filepath}, nodes: {len(self.node_dict)}")
            return True
            
        except Exception as e:
            print(f"Load error: {filepath}, {e}")
            return False
    
    @classmethod
    def load_trained_agent(cls, filepath, game, agent_id):
        agent = cls(game, agent_id)
        agent.load_agent(filepath, game)
        return agent

    def _convert_nodes_to_cpu(self):
        for obs, node in self.node_dict.items():
            if node.use_gpu:
                node.use_gpu = False
                node.cum_regrets = node.cum_regrets.cpu().numpy()
                node.curr_policy = node.curr_policy.cpu().numpy()
                node.sum_policy = node.sum_policy.cpu().numpy()
                node.learned_policy = node.learned_policy.cpu().numpy()

    def get_performance_info(self):
        info = {
            'num_nodes': len(self.node_dict),
            'max_depth': self.max_depth
        }
        
        return info


