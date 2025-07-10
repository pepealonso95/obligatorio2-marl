import random
from itertools import product
from gymnasium.spaces import Discrete, Tuple
from base.game import AlternatingGame, AgentID, ActionType
from games.nocca_nocca.board import Board, MOVES, MAX_STACK, ROWS, COLS
from games.nocca_nocca.board import Player, BLACK, WHITE
from games.nocca_nocca.board import Action

class NoccaNocca(AlternatingGame):
    def __init__(self, initial_player=None, max_steps=None, seed=None, render_mode='human'):
        super().__init__()

        self.metadata = {
            "name": "nocca_nocca_v0",
            "render_mode": "human",
            "agents": ["black", "white"],
            "agent_order": "random"
        }

        self.render_mode = render_mode
        self.initial_player = initial_player
        self.max_steps = max_steps
        self.seed = seed
        random.seed(seed)

        self.board = None

        self.agents: list[AgentID] = ["Black", "White"]
        self.players: list[Player] = [BLACK, WHITE]
        self.n_agents = len(self.agents)
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.n_agents))))
        self.agent_selection = None

        self.action_board_dict = dict(map(lambda x: x, enumerate((product(range(ROWS), range(COLS), MOVES)))))
        self.board_action_dict = dict(map(lambda x: (x[1], x[0]), self.action_board_dict.items()))
        self.action_list = list(self.action_board_dict.keys())
        self.n_actions = len(self.action_list)
        self.action_spaces = {agent: Discrete(self.n_actions) for agent in self.agents}

        self.observation_spaces = {agent: Discrete(self.n_actions) for agent in self.agents}

    def available_actions(self) -> list[ActionType]:
        player = self.agent_name_mapping[self.agent_selection]
        board_actions = self.board.legal_moves(player=player)
        actions = list(map(lambda x: self.board_action_dict[x], board_actions))
        return actions
    
    def step(self, action: ActionType) -> None:
        
        if self.terminated():
            raise ValueError(f"Game has already finished - Call reset() if you want to play again")
    
        player = self.agent_name_mapping[self.agent_selection]
        board_action = self.action_board_dict[action]

        valid_action, message = self.board.is_legal_move(player=player, action=board_action)
        if not valid_action:
            raise ValueError(f"Invalid board action {board_action} ({action}) for agent {self.agent_selection}({player}) - {message}.")
        
        self.board.play_turn(player=player, action=board_action)

        self.steps += 1

        _game_over = self.board.check_game_over()
        _truncated = self._check_truncated()
        if _game_over or _truncated:
            self.terminations = dict(map(lambda agent: (agent, True), self.agents))
            self.truncations = dict(map(lambda agent: (agent, _truncated), self.agents))
            self._set_rewards()
        else:
            next_player = self.board._opponent(player=player)
            self.agent_selection = self.agents[next_player]

        self.observations = dict(map(lambda agent: (agent, self.board.squares), self.agents))
        self.infos = dict(map(lambda agent: (agent, {}), self.agents))

    def _check_truncated(self):
        return (self.max_steps is not None and self.steps >= self.max_steps)

    def _set_rewards(self):
        winner = self.board.check_for_winner()
        if winner is not None:
            for p in self.players:
                agent = self.agents[p]
                if p == winner:
                    self.rewards[agent] = 1
                else:
                    self.rewards[agent] = -1
        else:
            for agent in self.agents:
                self.rewards[agent] = 0

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self.board = Board()

        if self.initial_player is None:
            self.agent_selection = self.agents[random.choice(self.players)]
        else:
            self.agent_selection = self.agents[self.initial_player]

        self.steps = 0

        self.observations = dict(map(lambda agent: (agent, self.board.squares), self.agents))
        self.rewards = dict(map(lambda agent: (agent, 0), self.agents))
        self.terminations = dict(map(lambda agent: (agent, False), self.agents))
        self.truncations = dict(map(lambda agent: (agent, False), self.agents))
        self.infos = dict(map(lambda agent: (agent, {}), self.agents))

    def render(self):
        self.board.render()

    def check_for_winner(self):
        winner = self.board.check_for_winner()
        if winner is not None:
            return self.agents[winner]
        else:
            return None
    
    def clone(self):
        return super().clone()
    
    def observe(self, agent: AgentID):
        """Observación hashable para CFR"""
        if self.board is None:
            return "initial_state"
        return str(self.board.squares)
    
    def eval(self, agent: AgentID) -> float:
        if agent not in self.agents:
            raise ValueError(f"Agent {agent} is not part of the game.")

        if self.terminated():
            return self.rewards[agent]
    
        player = self.agent_name_mapping[agent]
        return 0. * player
