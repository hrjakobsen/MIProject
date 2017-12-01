from games.pong import *

class GreedyPongAgent(object):
    def __init__(self, player):
        self.player = player

    def finalize(self, state, reward):
        pass

    def getMove(self, state: PongGame, reward, actions):
        pos = state.p1pos if self.player == 1 else state.p2pos
        return actions[1] if state.ballPosition[1] > pos else actions[2]
