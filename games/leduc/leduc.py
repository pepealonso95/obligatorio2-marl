import numpy as np
from numpy import random
from gymnasium.spaces import Discrete, Text, Dict
from base.game import AlternatingGame, AgentID, ActionType

class LeducPoker(AlternatingGame):

    def __init__(self, initial_player=None, seed=None, render_mode='human'):
        self.render_mode = render_mode
        self.seed = seed
        random.seed(seed)

        self.agents = ["agent_0", "agent_1"]
        self.players = [0, 1]
        self.initial_player = initial_player
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self.agent_selection = None

        # 0=fold, 1=call, 2=raise
        self._moves = ['f', 'c', 'r']
        self._num_actions = 3
        self.action_spaces = {
            agent: Discrete(self._num_actions) for agent in self.agents
        }

        # 6 cartas: 2 de cada tipo
        self._card_names = ['J', 'Q', 'K']
        self._deck = [0, 0, 1, 1, 2, 2]
        self._hand = None
        self._public_card = None
        
        self._hist = None
        self._round = 1  # 1 o 2
        self._pot = [1, 1]  # Ante
        self._max_raises_per_round = 2
        self._raises_this_round = 0
        
        self._terminal_round1 = set(['fc', 'ff', 'cc', 'crc', 'crf', 'rcf', 'rcc', 'rrc', 'rrf'])
        self._terminal_round2 = set()
        
        self.observation_spaces = {
            agent: Dict({
                'hand': Discrete(3),  # J=0, Q=1, K=2
                'public': Discrete(4),  # -1=none, 0=J, 1=Q, 2=K
                'hist': Text(min_length=0, max_length=10, charset=frozenset(self._moves)),
                'round': Discrete(2),  # 1 o 2
                'pot': Discrete(20)
            }) for agent in self.agents
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            
        # Repartir cartas
        deck = self._deck.copy()
        random.shuffle(deck)
        self._hand = {self.agents[0]: deck[0], self.agents[1]: deck[1]}
        self._public_card = deck[2]  # Se revela después de ronda 1
        
        self._hist = ''
        self._round = 1
        self._pot = [1, 1]  # Ante
        self._raises_this_round = 0
        
        if self.initial_player is None:
            self.agent_selection = self.agents[0]
        else:
            self.agent_selection = self.agents[self.initial_player]
            
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def step(self, action: ActionType) -> None:
        if self.game_over():
            raise ValueError("Game has already finished")
            
        current_player = self.agent_name_mapping[self.agent_selection]
        move = self._moves[action]
        self._hist += move
        
        if move == 'f':  # fold
            self._handle_fold()
        elif move == 'r':  # raise
            self._handle_raise(current_player)
        
        if self._is_round_terminal():
            if self._round == 1:
                self._start_round_2()
            else:
                self._end_game()
        else:
            self.agent_selection = self.agents[1 - current_player]

    def _handle_fold(self):
        # Jugador actual se retira, el otro gana
        current_player = self.agent_name_mapping[self.agent_selection]
        other_player = 1 - current_player
        
        total_pot = sum(self._pot)
        self.rewards[self.agents[other_player]] = total_pot - self._pot[other_player]
        self.rewards[self.agents[current_player]] = -self._pot[current_player]
        
        for agent in self.agents:
            self.terminations[agent] = True

    def _handle_raise(self, player):
        if self._raises_this_round >= self._max_raises_per_round:
            # Tratar como call si se alcanzó el máximo de raises
            return
            
        self._pot[player] += 1
        self._raises_this_round += 1

    def _is_round_terminal(self):
        if self._round == 1:
            return self._hist in self._terminal_round1
        else:
            round2_hist = self._hist[len(self._hist.split('|')[0]) + 1:] if '|' in self._hist else self._hist
            return len(round2_hist) >= 2 and self._is_betting_complete(round2_hist)

    def _is_betting_complete(self, hist):
        if 'f' in hist:
            return True
        if hist.endswith('cc'):
            return True
        if hist.count('r') >= 2 and hist.endswith('c'):
            return True
        return False

    def _start_round_2(self):
        self._round = 2
        self._raises_this_round = 0
        self._hist += '|'  # Separador entre rondas
        
        self.agent_selection = self.agents[0]

    def _end_game(self):
        # Showdown - comparar manos con carta pública
        hand0 = self._hand[self.agents[0]]
        hand1 = self._hand[self.agents[1]]
        public = self._public_card
        
        # Verificar pares
        pair0 = (hand0 == public)
        pair1 = (hand1 == public)
        
        if pair0 and not pair1:
            winner = 0
        elif pair1 and not pair0:
            winner = 1
        elif pair0 and pair1:
            # Ambos tienen pares, carta más alta gana
            winner = 0 if hand0 > hand1 else 1
        else:
            # Sin pares, carta más alta gana
            winner = 0 if hand0 > hand1 else 1
            
        total_pot = sum(self._pot)
        self.rewards[self.agents[winner]] = total_pot - self._pot[winner]
        self.rewards[self.agents[1-winner]] = -self._pot[1-winner]
        
        for agent in self.agents:
            self.terminations[agent] = True

    def observe(self, agent: AgentID):
        """Observación para CFR"""
        player_idx = self.agent_name_mapping[agent]
        hand = self._hand[agent]
        
        # Formato: "mano|publica|historia|ronda"
        public_str = str(self._public_card) if self._round == 2 else "None"
        obs_str = f"{hand}|{public_str}|{self._hist}|{self._round}"
        
        return obs_str

    def available_actions(self):
        """Acciones disponibles (0=fold, 1=call, 2=raise)"""
        if self.game_over():
            return []
            
        actions = [0, 1]  # fold, call siempre disponibles
        
        if self._raises_this_round < self._max_raises_per_round:
            actions.append(2)
            
        return actions

    def game_over(self):
        return any(self.terminations.values())

    def reward(self, agent: AgentID):
        return self.rewards.get(agent, 0)

    def clone(self):
        """Crear copia del estado del juego"""
        new_game = LeducPoker(initial_player=self.initial_player, seed=self.seed)
        new_game._hand = self._hand.copy() if self._hand else None
        new_game._public_card = self._public_card
        new_game._hist = self._hist
        new_game._round = self._round
        new_game._pot = self._pot.copy() if self._pot else None
        new_game._raises_this_round = self._raises_this_round
        new_game.agent_selection = self.agent_selection
        new_game.rewards = self.rewards.copy() if self.rewards else {}
        new_game.terminations = self.terminations.copy() if self.terminations else {}
        new_game.truncations = self.truncations.copy() if self.truncations else {}
        new_game.infos = self.infos.copy() if self.infos else {}
        return new_game

    def render(self):
        if self.render_mode == 'human':
            print(f"Round {self._round}, History: {self._hist}")
            print(f"Pot: {self._pot}, Public: {self._card_names[self._public_card] if self._round == 2 else 'Hidden'}")
            print(f"Current player: {self.agent_selection}")
            if self.game_over():
                print(f"Rewards: {self.rewards}")
