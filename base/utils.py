from base.game import AlternatingGame
from base.agent import Agent, AgentID
import numpy as np

def play(game: AlternatingGame, agents: dict[AgentID, Agent]):
    game.reset()
    game.render()
    while not game.terminated():
        action = agents[game.agent_selection].action()
        game.step(action)
        game.render()
    print(game.rewards)

def run(game: AlternatingGame, agents: dict[AgentID, Agent], N=100, verbose=False):
    values = []
    for i in range(1, N):    
        if verbose:
            print(f"Game {i}/{N}")
        game.reset()
        while not game.terminated():
            action = agents[game.agent_selection].action()
            game.step(action)
        if verbose:
            for agent in game.agents:
                print(f"Reward {agent}: {game.reward(agent)}")
        values.append(game.reward(game.agents[0]))
    v, c = np.unique(values, return_counts=True)
    return dict(zip(v, c)), np.mean(values)
