from interfaces import IAgent
import pygame
import copy
import numpy as np


class Trainer(object):
    def __init__(self, agent1 : IAgent, agent2 : IAgent, gameFunction, epsilon=0.1, playSeed=0, trainSeed=0):
        self.agent1 = agent1
        self.agent2 = agent2
        self.gameFunction = gameFunction
        self.epsilon = epsilon
        self.playSeed = playSeed
        self.trainSeed = trainSeed
        self.resolution = (1600, 800)
        self.recentOutcomes = None

    def run(self, numPlays, numTrains, numRepeatGames, verbose=False, visualise=False):
        print("Starting full run: {0} test games, {1} training games, {0} test games".format(numPlays, numTrains))
        self.play(numPlays, verbose, visualise)
        if verbose:
            p1Wins, p2Wins, draw = self.countOutcomes()
            print("Player 1: {} | Player 2: {} | Draws: {}".format(p1Wins, p2Wins, draw))
        print("Finished {} test games".format(numPlays))

        self.train(numTrains, numRepeatGames, verbose)
        print("Finished {} training games".format(numTrains))

        self.play(numPlays, verbose, visualise)
        if verbose:
            p1Wins, p2Wins, draw = self.countOutcomes()
            print("Player 1: {} | Player 2: {} | Draws: {}".format(p1Wins, p2Wins, draw))
        print("Finished {} test games".format(numPlays))

    def train(self, numGames, numRepeatGames, verbose):
        np.random.seed(self.playSeed)
        interval = max(numGames / 100, 1)

        for x in range(numGames):
            if x % numRepeatGames == 0:
                startGame = self.gameFunction()
            game = copy.deepcopy(startGame)

            while not game.gameEnded():
                if game.getTurn() == 1:
                    makeMove(self.agent1, game, 1, self.epsilon)
                else:
                    makeMove(self.agent2, game, 2, self.epsilon)

            self.agent1.finalize(game)
            self.agent2.finalize(game)

            if verbose and x % interval == 0:
                print("\rTrained %s/%s games - Agent 1: %s - Agent 2: %s" % (x, numGames, self.agent1.getInfo(), self.agent2.getInfo()), end="")
        if verbose:
            print()

    def play(self, numGames, verbose=False, visualise=False):
        np.random.seed(self.playSeed)
        self.recentOutcomes = []
        interval = max(numGames / 100, 1)
        draw = True

        if visualise:
            pygame.init()
            pygame.display.set_mode(self.resolution)
            surface = pygame.display.get_surface()

        for x in range(numGames):
            game = self.gameFunction()

            if visualise:
                p1Wins, p2Wins, draws = self.countOutcomes()
                pygame.display.set_caption("Player 1: {} | Player 2: {} | Draws: {}".format(p1Wins, p2Wins, draws))

            while not game.gameEnded():
                if visualise:
                    if draw:
                        game.draw(surface)
                        pygame.display.flip()

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_t:
                                draw = not draw
                            elif event.key == pygame.K_q:
                                pygame.quit()
                                return

                if game.getTurn() == 1:
                    game.makeMove(1, self.agent1.getTrainedMove(game))
                else:
                    game.makeMove(2, self.agent2.getTrainedMove(game))
            self.recentOutcomes.append(game.getWinner())

            if verbose and x % interval == 0:
                print("\rPlayed %s/%s games" % (x + 1, numGames), end="")
        if verbose:
            print()

        if visualise:
            pygame.quit()

    def countOutcomes(self):
        p1Wins = len([g for g in self.recentOutcomes if g == 1])
        p2Wins = len([g for g in self.recentOutcomes if g == 2])
        draws = len([g for g in self.recentOutcomes if g == -1])
        return p1Wins, p2Wins, draws


def makeMove(agent: IAgent, game, player, epsilon):
    action = agent.getMove(game)
    if np.random.rand() < epsilon:
        actions = game.getActions(player)
        action = actions[np.random.randint(len(actions))]
        agent.s = None
    game.makeMove(player, action)
