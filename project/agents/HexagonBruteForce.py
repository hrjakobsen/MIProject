from games.HexagonGame import HexagonGame
import numpy as np
import copy
import sys

sys.setrecursionlimit(1000000)

np.random.seed(0)

gamma = 1
player = 1
width = 3
height = 3

game = HexagonGame(width, height)
actions = game.getActions()
Q = {}

def calculateQ(state : HexagonGame, action):
    nextState = copy.deepcopy(state)
    nextState.makeMove(player, action)

    if nextState.gameEnded():
        return

    maxQ = -9999999
    for a in actions:
        calculateQ(nextState, a)
        tempQ = Q[(nextState.hash, a)]
        if tempQ > maxQ:
            maxQ = tempQ

    Q[(state.hash(), action)] = state.getReward(player) + gamma * maxQ

calculateQ(game, actions[0])

print(Q)
