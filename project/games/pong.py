from interfaces import IGame
from interface import implements
import numpy as np
import copy
import math
import pygame
import pygame.gfxdraw

UP = 1
NOTHING = 0
DOWN = -1

BACKGROUNDCOLOR = (255, 255, 255)
FRAMECOLOR = (0, 0, 0)
P1COLOR = (30, 160, 0)
P2COLOR = (0, 30, 255)
BALLCOLOR = (255, 0, 0)
MBFRAMES = 6
PADDLEWIDTH = 12

class Pong(implements(IGame)):
    def __init__(self, width=100, height=50, ballVelocity=None, ballPosition=None):
        self.width = width
        self.height = height
        self.ballPosition = np.array([width // 2, height // 2]) if ballPosition is None else ballPosition
        self.p1pos = self.height // 2
        self.p2pos = self.height // 2
        self.p1Bounces = 0
        self.p2Bounces = 0
        self.ballRadius = 1
        self.paddleSpeed = 0.5
        self.paddleHeight = 15
        self.ballPositions = []
        self.winner = None
        self.numFeatures = None
        self.lastP1Action = None
        self.lastP2Action = None
        self.actions = [NOTHING, UP, DOWN]
        self.turn = self.turn = np.random.randint(2) + 1
        self.ballVelocity = ballVelocity
        if ballVelocity is None:
            direction = np.random.randint(180)
            if direction < 90:
                direction -= 45
            else:
                direction += 45

            dirRad = direction * math.pi / 180
            self.ballVelocity = np.array([math.cos(dirRad), math.sin(dirRad)])

    def __deepcopy__(self, _):
        new = Pong(self.width, self.height, self.ballVelocity, self.ballPosition)
        new.p1pos = self.p1pos
        new.p2pos = self.p2pos
        new.p1Bounces = self.p1Bounces
        new.p2Bounces = self.p2Bounces
        new.ballRadius = self.ballRadius
        new.paddleSpeed = self.paddleSpeed
        new.paddleHeight = self.paddleHeight
        new.winner = self.winner
        new.numFeatures = self.numFeatures
        new.lastP1Action = self.lastP1Action
        new.lastP2Action = self.lastP2Action
        new.turn = self.turn

        return new

    def getActions(self, player):
        return self.actions

    def getTurn(self):
        return self.turn

    def getNumFeatures(self):
        if self.numFeatures is None:
            self.numFeatures = len(calculateFeatures(self, 0, 1))

        return self.numFeatures

    def getFeatures(self, player, action):
        return calculateFeatures(self, action, player)

    def makeMove(self, player, action):
        self.turn = self.turn % 2 + 1
        if (player == 1 and self.lastP1Action is not None) or (player == 2 and self.lastP2Action is not None):
            raise ValueError("Player {} cannot make two actions in a row".format(player))
        if player == 1:
            self.lastP1Action = action
        else:
            self.lastP2Action = action

        if self.lastP1Action is None or self.lastP2Action is None:
            return self

        tempState = simulateMove(self, self.lastP1Action, self.lastP2Action)
        self.p1pos = tempState.p1pos
        self.p2pos = tempState.p2pos
        self.ballPosition = tempState.ballPosition
        self.ballVelocity = tempState.ballVelocity
        self.winner = tempState.winner
        self.p1Bounces = tempState.p1Bounces
        self.p2Bounces = tempState.p2Bounces

        self.lastP1Action, self.lastP2Action = None, None

    def gameEnded(self):
        return self.winner is not None

    def getReward(self, player):
        if player == 1 and self.p1Bounces == 1:
            self.p1Bounces = 0
            return 1
        elif player == 2 and self.p2Bounces == 1:
            self.p2Bounces = 0
            return 1
        return 0

    def getWinner(self):
        return self.winner

    def draw(self, surface):
        surface.fill((0, 0, 0))
        sizeModifier = min((surface.get_width() - PADDLEWIDTH * 2) / self.width, surface.get_height() / self.height)

        pygame.gfxdraw.box(surface, (0, 0, self.width * sizeModifier + PADDLEWIDTH * 2, self.height * sizeModifier), BACKGROUNDCOLOR)

        # Center line
        pygame.gfxdraw.vline(surface, int(self.width * sizeModifier) // 2, 0, int(self.height * sizeModifier), FRAMECOLOR)

        # Top and bottom lines
        pygame.gfxdraw.box(surface, (0, 0, int(self.width * sizeModifier + 2 * PADDLEWIDTH), 2), FRAMECOLOR)
        pygame.gfxdraw.box(surface, (0, int(self.height * sizeModifier), int(self.width * sizeModifier + 2 * PADDLEWIDTH), 2), FRAMECOLOR)

        # Paddle 1
        p1Paddle = (0, int(self.p1pos - self.paddleHeight / 2) * sizeModifier, PADDLEWIDTH, self.paddleHeight * sizeModifier)
        pygame.gfxdraw.box(surface, p1Paddle, P1COLOR)

        # Paddle 2
        p2Paddle = (self.width * sizeModifier + PADDLEWIDTH, int(self.p2pos - self.paddleHeight / 2) * sizeModifier, PADDLEWIDTH, self.paddleHeight * sizeModifier)
        pygame.gfxdraw.box(surface, p2Paddle, P2COLOR)

        # Ball
        ballPos = self.ballPosition * sizeModifier
        self.ballPositions.append(ballPos)

        for i, position in enumerate(self.ballPositions):
            col = interpolateRGB(BACKGROUNDCOLOR, BALLCOLOR, (i + 1) / len(self.ballPositions))
            pygame.gfxdraw.aacircle(surface, int(position[0] + PADDLEWIDTH), int(position[1]), int(self.ballRadius * sizeModifier), col)
            pygame.gfxdraw.filled_circle(surface, int(position[0] + PADDLEWIDTH), int(position[1]), int(self.ballRadius * sizeModifier), col)

        if len(self.ballPositions) >= MBFRAMES:
            self.ballPositions.pop(0)

        pygame.time.delay(600)


def interpolateRGB(color1, color2, t):
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    return r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t


def simulatePlayerMove(state, action, player):
    return simulateMove(state, action, NOTHING) if player == 1 else simulateMove(state, NOTHING, action)


def simulateMove(state, action1, action2):
    newState = copy.deepcopy(state)

    newState.p1pos = paddleUpdate(state, action1, 1)
    newState.p2pos = paddleUpdate(state, action2, 2)
    newState.ballPosition, newState.ballVelocity, newState.winner, bounced = updateBall(state)

    if bounced:
        if newState.ballPosition[0] < newState.width // 2:
            newState.p1Bounces += 1
        else:
            newState.p2Bounces += 1

    return newState


def paddleUpdate(state, action, player):
    myPos = state.p1pos if player == 1 else state.p2pos
    return max(min(state.height - state.paddleHeight // 2, myPos + state.paddleSpeed * action), state.paddleHeight // 2)


def updateBall(state):
    bounced = False
    newVel = state.ballVelocity.copy()
    newPos = state.ballPosition.copy() + newVel

    factorPaddle, factorWall = factorToPaddle(state), factorToWall(state)

    if factorPaddle > 1 and factorWall > 1:
        return newPos, state.ballVelocity, None, bounced

    if factorWall < 1 and factorPaddle < 1:
        # Update the smallest factor first
        if factorWall < factorPaddle:
            newPos = state.ballPosition + state.ballVelocity * factorWall
            newVel[1] *= -1
            factorPaddle -= factorWall
            # We are now at the wall
            # We should now move to x = 0 or x = width
            newPos = newPos + newVel * factorPaddle

            # We can now check if ball is bouncing off of paddle
            paddlePos = state.p1pos if (newPos[0] < state.width / 2) else state.p2pos
            if paddlePos - state.paddleHeight // 2 < newPos[1] < paddlePos + state.paddleHeight // 2:
                # Bounced off wall and then paddle!
                bounced = True
                newVel[0] *= -1
                remainingFactor = 1 - (factorPaddle + factorWall)
                newPos += newVel * remainingFactor
            else:
                # Ball bounced off wall and someone lost
                remainingFactor = 1 - (factorPaddle + factorWall)
                newPos += newVel * remainingFactor
                return newPos, newVel, int(newPos[0] < state.width // 2) + 1, bounced
        else:
            newPos = state.ballPosition + state.ballVelocity * factorPaddle
            paddlePos = state.p1pos if (newPos[0] < state.width / 2) else state.p2pos
            if paddlePos - state.paddleHeight // 2 <= newPos[1] <= paddlePos + state.paddleHeight // 2:
                # Bounced off paddle and then wall!
                bounced = True
                newVel[0] *= -1
                newPos += state.ballVelocity * (factorPaddle - factorWall)
                newVel[1] *= -1
                newPos += state.ballVelocity * (1 - factorPaddle - factorWall)
            else:
                # Someone lost
                remainingFactor = 1 - factorPaddle
                newPos += newVel * remainingFactor
                return newPos, newVel, int(newPos[0] < state.width // 2) + 1, bounced

    elif factorWall < 1:
        newPos = state.ballPosition + state.ballVelocity * factorWall
        newVel = state.ballVelocity * np.array([1, -1])
        newPos += state.ballVelocity * (1 - factorWall)
    elif factorPaddle < 1:
        newPos = state.ballPosition + state.ballVelocity * factorPaddle
        paddlePos = state.p1pos if (newPos[0] < state.width / 2) else state.p2pos
        if paddlePos - state.paddleHeight // 2 < newPos[1] < paddlePos + state.paddleHeight // 2:
            # Bounced off paddle!
            bounced = True
            newVel[0] *= -1
            remainingFactor = 1 - factorPaddle
            newPos += newVel * remainingFactor
        else:
            # Someone lost
            remainingFactor = 1 - factorPaddle
            newPos += newVel * remainingFactor
            return newPos, newVel, int(newPos[0] < state.width // 2) + 1, bounced

    return newPos, newVel, None, bounced


def factorToWall(state):
    """
    this function returns a factor that multiplied to the velocity vector results in
    hitting the top or bottom of the board depending on direction
    :param state:
    :return:
    """
    if state.ballVelocity[1] == 0:
        return 1e20  # a large value
    return max([(wall - state.ballPosition[1]) / state.ballVelocity[1] for wall in
                [state.ballRadius, state.height - state.ballRadius]])


def factorToPaddle(state):
    """
    this function returns a factor that multiplied to the velocity vector results in
    hitting the left side or right side of the board depending on direction
    :param state:
    :return:
    """

    return max(
        (state.ballRadius - state.ballPosition[0]) / state.ballVelocity[0],
         (state.width - state.ballRadius - state.ballPosition[0]) / state.ballVelocity[0]
        )


def calculateFeatures(state, action, player):
    newState = simulatePlayerMove(state, action, player)

    results = np.array([
        1,
        getAngle(newState, player),
        distanceFromCenter(newState, player)
    ])

    return results


def getAngle(state, player):
    #return 0 if the ball is not travelling in the direction of the player
    if player == 1 and state.ballVelocity[0] > 0:
        return 0
    if player == 2 and state.ballVelocity[0] < 0:
        return 0

    vector = getVectorBetweenBallAndPaddle(state, player)
    lengthOfPaddleVec = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1])

    #avoid division by 0
    lengthOfPaddleVec = 0.000001 if lengthOfPaddleVec == 0 else lengthOfPaddleVec
    dotProduct = np.dot(vector, state.ballVelocity)
    angle = math.acos(dotProduct / np.dot(lengthOfPaddleVec, math.sqrt(state.ballVelocity[0] * state.ballVelocity[0] + state.ballVelocity[1] * state.ballVelocity[1])))

    return angle


def getVectorBetweenBallAndPaddle(state, player):
    playerX = 0 if player == 1 else state.width
    playerY = state.p1pos if player == 1 else state.p2pos
    return np.array([playerX - state.ballPosition[0], playerY - state.ballPosition[1]])


def distanceFromCenter(state, player):
    dist = abs(state.p1pos - state.height // 2) if player == 1 else abs(state.p2pos - state.height // 2)
    return dist // state.height


def distanceToBall(state, player):
    paddleY = state.p1pos if player == 1 else state.p2pos
    return (state.ballPosition[1] - paddleY) ** 2
