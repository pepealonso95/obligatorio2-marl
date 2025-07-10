import numpy as np
from numpy import random
from gymnasium.spaces import Discrete, Text, Dict
from base.game import AlternatingGame, AgentID, ActionType

class KuhnPoker3Player(AlternatingGame):

    def __init__(self, initial_player=None, seed=None, render_mode='human'):
        self.render_mode = render_mode
        self.seed = seed
        random.seed(seed)

        self.agents = ["agent_0", "agent_1", "agent_2"]
        self.players = [0, 1, 2]
        self.initial_player = initial_player
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self.agent_selection = None

        self._moves = ['p', 'b']
        self._num_actions = 2
        self.action_spaces = {
            agent: Discrete(self._num_actions) for agent in self.agents
        }

        # 3 cartas para 3 jugadores
        self._card_names = ['J', 'Q', 'K']
        self._num_cards = len(self._card_names)
        self._cards = list(range(self._num_cards))
        self._card_space = Discrete(self._num_cards)
        self._hand = None

        self._hist = None
        self._player = None
        self._active_players = None  
        self._pot = 3 
        self._bets = None  
        
        self.observation_spaces = {
            agent: Dict({
                'card': self._card_space,
                'hist': Text(min_length=0, max_length=10, charset=frozenset(self._moves))
            }) for agent in self.agents
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            
        # Repartir cartas (las 3 cartas a 3 jugadores)
        deck = self._cards.copy()
        random.shuffle(deck)
        self._hand = deck 
        
        self._hist = ''
        self._active_players = set([0, 1, 2])
        self._bets = [1, 1, 1] 
        
        if self.initial_player is None:
            self._player = 0
        else:
            self._player = self.initial_player
            
        self.agent_selection = self.agents[self._player]
        
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def step(self, action: ActionType) -> None:
        if self.game_over():
            raise ValueError("Game has already finished")
            
        current_player = self._player
        move = self._moves[action]
        self._hist += move
        
        if move == 'p':  
            # Jugador pasa - permanece en el juego pero no aumenta apuesta
            pass
        elif move == 'b':  # bet
            # Jugador apuesta - incrementa su contribución
            self._bets[current_player] += 1
            
        if self._should_end_game():
            self._end_game()
        else:
            self._next_player()

    def _should_end_game(self):
        """Verificar si el juego debe terminar basado en reglas de 3 jugadores"""
        # Contar acciones recientes para determinar si la ronda de apuestas está completa
        recent_actions = self._hist[-3:] if len(self._hist) >= 3 else self._hist
        
        # Regla simple: si tenemos 3 acciones y no hay apuesta reciente, o si alguien se retiró
        if len(self._active_players) == 1:
            return True 

        if len(self._hist) >= 3:
            if len(recent_actions) == 3 and 'b' not in recent_actions:
                return True
            if 'b' in self._hist and len(self._hist) >= 6:
                return True
                
        return False

    def _next_player(self):
        """Mover al siguiente jugador activo"""
        original_player = self._player
        while True:
            self._player = (self._player + 1) % 3
            if self._player in self._active_players:
                break
            if self._player == original_player: 
                break
                
        self.agent_selection = self.agents[self._player]

    def _end_game(self):
        """Terminar el juego y calcular pagos"""
        if len(self._active_players) == 1:
            # Solo queda un jugador - ese gana
            winner = list(self._active_players)[0]
            total_pot = sum(self._bets)
            for i, agent in enumerate(self.agents):
                if i == winner:
                    self.rewards[agent] = total_pot - self._bets[i]
                else:
                    self.rewards[agent] = -self._bets[i]
        else:
            # Showdown - carta más alta gana
            active_hands = {p: self._hand[p] for p in self._active_players}
            winner = max(active_hands.keys(), key=lambda p: active_hands[p])
            
            total_pot = sum(self._bets)
            for i, agent in enumerate(self.agents):
                if i == winner:
                    self.rewards[agent] = total_pot - self._bets[i]
                else:
                    self.rewards[agent] = -self._bets[i]
        
        for agent in self.agents:
            self.terminations[agent] = True

    def observe(self, agent: AgentID):
        """Observación para CFR"""
        player_idx = self.agent_name_mapping[agent]
        card = self._hand[player_idx]
        
        # Conjunto de información: carta + historial de acciones
        obs_str = f"{card}|{self._hist}"
        return obs_str

    def available_actions(self):
        """Acciones disponibles"""
        if self.game_over():
            return []
        return [0, 1] 

    def game_over(self):
        return any(self.terminations.values())

    def reward(self, agent: AgentID):
        return self.rewards.get(agent, 0)

    def clone(self):
        """Crear copia del estado del juego"""
        new_game = KuhnPoker3Player(initial_player=self.initial_player, seed=self.seed)
        new_game._hand = self._hand.copy() if self._hand else None
        new_game._hist = self._hist
        new_game._player = self._player
        new_game._active_players = self._active_players.copy() if self._active_players else None
        new_game._bets = self._bets.copy() if self._bets else None
        new_game.agent_selection = self.agent_selection
        new_game.rewards = self.rewards.copy() if self.rewards else {}
        new_game.terminations = self.terminations.copy() if self.terminations else {}
        new_game.truncations = self.truncations.copy() if self.truncations else {}
        new_game.infos = self.infos.copy() if self.infos else {}
        return new_game

    def render(self):
        if self.render_mode == 'human':
            print(f"History: {self._hist}")
            print(f"Hands: {[self._card_names[card] for card in self._hand]}")
            print(f"Bets: {self._bets}")
            print(f"Current player: {self.agent_selection}")
            if self.game_over():
                print(f"Rewards: {self.rewards}")
