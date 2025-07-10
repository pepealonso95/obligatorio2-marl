import copy
import pettingzoo.utils.env as env
from pettingzoo.utils.env import AECEnv

ObsType = env.ObsType
ObsDict = env.ObsDict
AgentID = env.AgentID
ActionType = env.ActionType
ActionDict = env.ActionDict

class AlternatingGame(AECEnv):

    observations: ObsDict
    rewards: dict[AgentID, float]
    terminations: dict[AgentID, bool]
    truncations: dict[AgentID, bool]
    infos: dict[AgentID, dict]

    agent_name_mapping: dict[AgentID, int]

    def observation_space(self, agent: AgentID):
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID):
        return self.action_spaces[agent]

    def num_actions(self, agent: AgentID):
        return self.action_space(agent).n
    
    def action_iter(self, agent: AgentID):
        return range(self.action_space(agent).n)
        
    def observe(self, agent: AgentID) -> ObsType:
        return self.observations[agent]
    
    def reward(self, agent: AgentID):
        return self.rewards[agent]
    
    def clone(self):
        game = copy.deepcopy(self)
        return game
    
    def done(self):
        return self.terminations[self.agent_selection]
    
    def terminated(self):
        return all([self.terminations[a] for a in self.agents])
    
    def truncated(self):
        return all([self.truncations[a] for a in self.agents])
    
    def game_over(self):
        return self.truncated() or self.terminated()
    
    def available_actions(self):
        pass



