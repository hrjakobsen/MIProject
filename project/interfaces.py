from interface import Interface


class IAgent(Interface):
    def getMove(self, state):
        pass

    def getTrainedMove(self, state):
        pass

    def finalize(self, state):
        pass

    def getInfo(self):
        pass


class IGame(Interface):
    def getActions(self, player):
        pass

    def getTurn(self):
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

    def getWinner(self):
        pass

    def draw(self, surface):
        pass
