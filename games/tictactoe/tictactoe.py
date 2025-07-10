from base.game import AgentID, ObsType
from numpy import ndarray
from gymnasium.spaces import Discrete, Text, Dict, Tuple
from pettingzoo.utils import agent_selector
from games.tictactoe import tictactoe_v3 as tictactoe
from base.game import AlternatingGame, AgentID
import numpy as np

import warnings
warnings.filterwarnings("ignore")

class TicTacToe(AlternatingGame):

    def __init__(self, render_mode=''):
        super().__init__()
        self.env = tictactoe.raw_env(render_mode=render_mode)
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces
        self.action_space = self.env.action_space
        self.agents = self.env.agents
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))

    def _update(self):
        self.rewards = self.env.rewards
        self.terminations = self.env.terminations
        self.truncations = self.env.truncations
        self.infos = self.env.infos
        self.agent_selection = self.env.agent_selection

    def reset(self):
        self.env.reset()
        self._update()

    def observe(self, agent: AgentID) -> ObsType:
        # Grid: 0=vacío, 1=agente, 2=oponente
        observation = self.env.observe(agent=agent)['observation']
        grid = np.sum(observation*[1,2], axis=2)
        # String para CFR
        return str(grid.flatten().tolist())

    def step(self, action):
        self.env.step(action)
        self._update()

    def available_actions(self):
        return self.env.board.legal_moves()

    def render(self):
        print("Player:", self.agent_selection)
        print("Board:") 
        sq = np.array(self.env.board.squares).reshape((3, 3))
        for i in range(3):
            for j in range(3):
                if sq[i, j] == 0:
                    print(" . ", end="")
                elif sq[i, j] == 1:
                    print(" X ", end="")
                else:
                    print(" O ", end="")
            print()
        print()

    def clone(self):
        cloned_game = TicTacToe(render_mode=self.env.render_mode if hasattr(self.env, 'render_mode') else '')
        
        cloned_game.env.board.squares = self.env.board.squares.copy()
        
        cloned_game.env.rewards = self.env.rewards.copy()
        cloned_game.env.terminations = self.env.terminations.copy()
        cloned_game.env.truncations = self.env.truncations.copy()
        cloned_game.env.infos = self.env.infos.copy()
        
        if hasattr(self.env, 'agent_selection'):
            cloned_game.env.agent_selection = self.env.agent_selection
        else:
            cloned_game.env.agent_selection = self.env.agents[0]
        
        cloned_game._update()
        
        return cloned_game

    def eval(self, agent: AgentID) -> float:
        if agent not in self.agents:
            raise ValueError(f"Agent {agent} is not part of the game.")

        if self.terminated():
            return self.rewards[agent]
    
        # Grid para evaluación
        observation = self.env.observe(agent=agent)['observation']
        grid = np.sum(observation*[1,2], axis=2)

        E_agent = self._eval(grid, 2)
        E_opponent = self._eval(grid, 1)
        v = (E_agent - E_opponent) / 8.0

        return v
    
    def _eval(self, grid, player) -> float:
        rows = 0
        for i in range(3):
            rows += int(all(grid[i] != player))

        cols = 0
        for i in range(3):
            cols += int(all(grid.T[i] != player))           

        diag1 = int(all(grid.diagonal() != player))
        diag2 = int(all(np.fliplr(grid).diagonal() != player))

        return (rows + cols + diag1 + diag2)

    
    
