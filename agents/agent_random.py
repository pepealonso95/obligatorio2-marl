from base.game import AlternatingGame, AgentID
from base.agent import Agent
import numpy as np

class RandomAgent(Agent):

    def __init__(self, game: AlternatingGame, agent: AgentID, seed=None) -> None:
        super().__init__(game=game, agent=agent)

    def action(self):
        return np.random.choice(self.game.available_actions())
    
    def policy(self):
        raise ValueError('RandomAgent: Not implemented')
    