from interfaces import IAgent
from interface import implements


class PongGreedy(implements(IAgent)):
    def __init__(self, player):
        self.player = player

    def getMove(self, state):
        pos = state.p1pos if self.player == 1 else state.p2pos
        actions = state.getActions(self.player)
        return actions[1] if state.ballPosition[1] > pos else actions[2]

    def getTrainedMove(self, state):
        return self.getMove(state)

    def finalize(self, state):
        pass

    def getInfo(self):
        return
