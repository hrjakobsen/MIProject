from games.pong import PongGame

class PongAgent(object):
    def __init__(self, player):
        self.player = player

    def getMove(self, state : PongGame, reward):
        player = state.p1pos if self.player == 1 else state.p2pos
        if player < state.ballPosition[1]:
            return 1
        return -1


    def finalize(self, state, reward):
        pass