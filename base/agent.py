from base.game import AlternatingGame, AgentID

class Agent():

    def __init__(self, game:AlternatingGame, agent: AgentID) -> None:
        self.game = game
        self.agent = agent

    def action(self):
        pass

    def policy(self):
        pass
    
