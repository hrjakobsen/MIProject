from interface import Interface


class IAgent(Interface):
    def getMove(self, state):
        """
        Ask the agent which action to take in
        given state and update agent if applicable
        :param state: the current game
        :return: the action to play in this state
        """
        pass

    def getTrainedMove(self, state):
        """
        Ask the agent which action to take in
        given state
        :param state: the current game
        :return: the action to play in this state
        """
        pass

    def finalize(self, state):
        """
        Update the agent if applicable
        :param state: current state of game
        """
        pass

    def getInfo(self):
        """
        :return: String describing relevant information
        about the agent
        """
        pass


class IGame(Interface):
    def getActions(self, player):
        """
        :param player: id of player to get actions for
        :return: actions available to player
        """
        pass

    def getTurn(self):
        """
        :return: id of player whose turn it is
        """
        pass

    def getNumFeatures(self):
        """
        :return: Number of features in this game
        """
        pass

    def getFeatures(self, player, action):
        """
        Calculate features for current state, given action and player
        :param player: id of player to calculate features for
        :param action: action to calculate feature for
        :return: vector with getNumFeatures() elements
        """
        pass

    def makeMove(self, player, action):
        """
        Mutate the game, by performing given action as given player
        :param player: id of player
        :param action: action to perform
        """
        pass

    def gameEnded(self):
        """
        :return: true if game is over, else false
        """
        pass

    def getReward(self, player):
        """
        :param player: id of player
        :return: reward awarded to given player
        """
        pass

    def getWinner(self):
        """
        :return: the winner of the game if any, else None
        """
        pass

    def draw(self, surface):
        """
        Render a visual version of the game
        :param surface: pygame surface to draw the game on
        """
        pass
