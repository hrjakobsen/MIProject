from interface import Interface


class IAgent(Interface):
    def getMove(self, state, reward):
        pass

    def getTrainedMove(self, state):
        pass

    def finalize(self, state, reward):
        pass


class IGame(Interface):
    def getActions(self, player):
        pass

    def getNumFeatures(self):
        pass

    def getFeatures(self, player, action):
        pass

    def makeMove(self, player, action):
        pass

    def gameEnded(self):
        pass

    def getReward(self, player):
        pass
