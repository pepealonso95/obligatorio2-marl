# noqa: D212, D415

import numpy as np
from gymnasium import spaces
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.classic.tictactoe.board import TTT_GAME_NOT_OVER, TTT_TIE, Board
from pettingzoo.utils import AgentSelector, wrappers

def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human"],
        "name": "tictactoe_v3",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(
        self, render_mode: str | None = None, screen_height: int | None = 1000
    ):
        super().__init__()
        EzPickle.__init__(self, render_mode, screen_height)
        self.board = Board()

        self.agents = ["X", "O"]
        self.possible_agents = self.agents[:]

        self.action_spaces = {i: spaces.Discrete(9) for i in self.agents}
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(3, 3, 2), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(9,), dtype=np.int8),
                }
            )
            for i in self.agents
        }

        self.rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        self.render_mode = render_mode
        self.screen_height = screen_height
        self.screen = None
        
    def observe(self, agent):
        board_vals = np.array(self.board.squares).reshape(3, 3)
        cur_player = self.possible_agents.index(agent)
        opp_player = (cur_player + 1) % 2

        observation = np.empty((3, 3, 2), dtype=np.int8)
        # Capa 0: piezas del jugador actual, Capa 1: piezas del oponente
        observation[:, :, 0] = np.equal(board_vals, cur_player + 1)
        observation[:, :, 1] = np.equal(board_vals, opp_player + 1)

        action_mask = self._get_mask(agent)

        return {"observation": observation, "action_mask": action_mask}

    def _get_mask(self, agent):
        action_mask = np.zeros(9, dtype=np.int8)

        # Máscara solo para el agente seleccionado
        if agent == self.agent_selection:
            for i in self.board.legal_moves():
                action_mask[i] = 1

        return action_mask

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # Acción: valor 0-8 indicando posición en el tablero
    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        self.board.play_turn(self.agents.index(self.agent_selection), action)

        status = self.board.game_status()
        if status != TTT_GAME_NOT_OVER:
            if status == TTT_TIE:
                pass
            else:
                winner = status  # TTT_PLAYER1_WIN o TTT_PLAYER2_WIN
                loser = winner ^ 1  # 0 -> 1; 1 -> 0
                self.rewards[self.agents[winner]] += 1
                self.rewards[self.agents[loser]] -= 1

            # Juego terminado
            self.terminations = {i: True for i in self.agents}

        self.agent_selection = self.agents[(self.agents.index(self.agent_selection) + 1) % 2]

        if self.render_mode == "human":
            self.render()

    def reset(self, seed=None, options=None):
        self.board.reset()

        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        # Selecciona primer agente
        self.agent_selection = self.agents[0]

        if self.render_mode == "human":
            self.render()

    def close(self):
        pass

    def render(self):
        raise NotImplementedError(
            "Rendering is not implemented for this environment."
        )
