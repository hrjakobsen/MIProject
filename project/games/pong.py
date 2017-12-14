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

P1COLOR = (30, 160, 0)
P2COLOR = (0, 30, 255)
BALLCOLOR = (255, 0, 0)
BACKGROUNDCOLOR = (200, 200, 200)
PADDLEWIDTH = 8

class Pong(implements(IGame)):
    def __init__(self, width=100, height= 50, ballVelocity=None, ballPosition=None):
        self.width = width
        self.height = height
        self.p1pos = self.height // 2
        self.p2pos = self.height // 2
        self.ballPosition = np.array([width // 2, height // 2]) if ballPosition is None else ballPosition
        self.ballRadius = 3
        self.paddleSpeed = 0.5
        self.paddleHeight = 15
        self.winner = None
        self.numFeatures = None
        self.lastP1Action = None
        self.lastP2Action = None
        self.actions = [NOTHING, UP, DOWN]
        self.turn = None
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
        new.turn = self.turn
        return new

    def getActions(self, player):
        return self.actions

    def getTurn(self):
        if self.turn is None:
            self.turn = np.random.randint(2) + 1

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

        self.p1pos = paddleUpdate(self, self.lastP1Action, 1)
        self.p2pos = paddleUpdate(self, self.lastP2Action, 2)

        self.ballPosition, self.ballVelocity, self.winner = updateBall(self)
        self.lastP1Action, self.lastP2Action = None, None

    def gameEnded(self):
        return self.winner is not None

    def getReward(self, player):
        return getReward(self, player)

    def getWinner(self):
        return self.winner

    def draw(self, surface):
        surface.fill((0, 0, 0))
        sizeModifier = min((surface.get_width() - PADDLEWIDTH * 2) / self.width, surface.get_height() / self.height)
        pygame.gfxdraw.box(surface, (0, 0, self.width * sizeModifier + PADDLEWIDTH * 2, self.height * sizeModifier), BACKGROUNDCOLOR)

        # Paddle 1
        p1Paddle = (0, int(self.p1pos - self.paddleHeight // 2) * sizeModifier, PADDLEWIDTH, self.paddleHeight * sizeModifier)
        pygame.gfxdraw.box(surface, p1Paddle, P1COLOR)

        # Paddle 2
        p2Paddle = (self.width * sizeModifier + PADDLEWIDTH, int(self.p2pos - self.paddleHeight // 2) * sizeModifier, PADDLEWIDTH, self.paddleHeight * sizeModifier)
        pygame.gfxdraw.box(surface, p2Paddle, P2COLOR)

        # Ball
        bX, bY = self.ballPosition * sizeModifier
        pygame.gfxdraw.aacircle(surface, int(bX + PADDLEWIDTH), int(bY), int(self.ballRadius * sizeModifier), BALLCOLOR)
        pygame.gfxdraw.filled_circle(surface, int(bX + PADDLEWIDTH), int(bY), int(self.ballRadius * sizeModifier), BALLCOLOR)
        pygame.time.delay(6)


def updateBall(state):
    newPos = state.ballPosition + state.ballVelocity
    newVel = state.ballVelocity

    factorPaddle, factorWall = factorToPaddle(state), factorToWall(state)

    if factorPaddle > 1 and factorWall > 1:
        return newPos, state.ballVelocity, None

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
                # Bounced!
                newVel[0] *= -1
                remainingFactor = 1 - (factorPaddle + factorWall)
                newPos += newVel * remainingFactor
            else:
                # Someone lost
                remainingFactor = 1 - (factorPaddle + factorWall)
                newPos += newVel * remainingFactor
                return newPos, newVel, int(newPos[0] < state.width // 2) + 1
        else:
            newPos = state.ballPosition + state.ballVelocity * factorPaddle
            paddlePos = state.p1pos if (newPos[0] < state.width / 2) else state.p2pos
            if paddlePos - state.paddleHeight // 2 < newPos[1] < paddlePos + state.paddleHeight // 2:
                # Bounced!
                newVel[0] *= -1
                newPos += state.ballVelocity * (factorPaddle - factorWall)
                newVel[1] *= -1
                newPos += state.ballVelocity * (1 - factorPaddle - factorWall)
            else:
                # Someone lost
                remainingFactor = 1 - factorPaddle
                newPos += newVel * remainingFactor
                return newPos, newVel, int(newPos[0] < state.width // 2) + 1

    elif factorWall < 1:
        newPos = state.ballPosition + state.ballVelocity * factorWall
        newVel = state.ballVelocity * np.array([1, -1])
        newPos += state.ballVelocity * (1 - factorWall)
    elif factorPaddle < 1:
        newPos = state.ballPosition + state.ballVelocity * factorPaddle
        paddlePos = state.p1pos if (newPos[0] < state.width / 2) else state.p2pos
        if paddlePos - state.paddleHeight // 2 < newPos[1] < paddlePos + state.paddleHeight // 2:
            # Bounced!
            newVel[0] *= -1
            remainingFactor = 1 - factorPaddle
            newPos += newVel * remainingFactor
        else:
            # Someone lost
            remainingFactor = 1 - factorPaddle
            newPos += newVel * remainingFactor
            return newPos, newVel, int(newPos[0] < state.width // 2) + 1

    return newPos, newVel, None


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


def makeMove(state, action1, action2):
    newState = copy.deepcopy(state)

    newState.p1pos = paddleUpdate(state, action1, 1)
    newState.p2pos = paddleUpdate(state, action2, 2)

    newState.ballPosition, newState.ballVelocity, newState.winner = updateBall(state)

    return newState


def paddleUpdate(state, action, player):
    myPos = state.p1pos if player == 1 else state.p2pos
    return max(min(state.height - state.paddleHeight // 2, myPos + state.paddleSpeed * action), state.paddleHeight // 2)


def distanceToBall(s, a, player):
    s2 = makePlayerMove(s, a, player)
    paddleY = s2.p1pos if player == 1 else s2.p2pos
    return math.log2(1 / max(((s2.ballPosition[1] - paddleY) ** 2), 0.00001))


def calculateFeatures(state, action, player):
    nextState = makePlayerMove(state, action, player)

    results = np.array([
        #1,
        #nextState.p1pos if player == 1 else nextState.p2pos,
        #nextState.ballPosition[0],
        #nextState.ballPosition[1],
        #nextState.ballVelocity[0],
        #nextState.ballVelocity[1],
        #distanceToBall(nextState, player),
        getAngle(nextState, player),
        #distanceFromCenter(nextState, player)
        #getAngleLookahead(nextState, player)
    ])

    return results


def getAngle(s, player):
    #return 0 if the ball is not travelling in the direction of the player
    if player == 1 and s.ballVelocity[0] > 0:
        return 0
    if player == 2 and s.ballVelocity[0] < 0:
        return 0

    vector = getVectorBetweenBallAndPaddle(s, player)
    lengthOfPaddleVec = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1])

    #avoid division by 0
    lengthOfPaddleVec = 0.000001 if lengthOfPaddleVec == 0 else lengthOfPaddleVec
    dotProduct = np.dot(vector, s.ballVelocity)
    angle = math.acos(dotProduct / np.dot(lengthOfPaddleVec, 1))

    angle = 2 * math.pi - angle if angle > 0.5 * math.pi else angle
    return angle / 0.5 * math.pi


def distanceFromCenter(s, player):
    dist = abs(s.p1pos - s.height // 2) if player == 1 else abs(s.p2pos - s.height // 2)
    return dist // s.height


def getVectorBetweenBallAndPaddle(s, player):
    playerX = 0 if player == 1 else s.width
    playerY = s.p1pos if player == 1 else s.p2pos
    return np.array([playerX - s.ballPosition[0], playerY - s.ballPosition[1]])


def distanceToBall(s, player):
    paddleY = s.p1pos if player == 1 else s.p2pos

    return (s.ballPosition[1] - paddleY) ** 2


def makePlayerMove(s, a, player):
    return makeMove(s, a, NOTHING) if player == 1 else makeMove(s, NOTHING, a)


def getReward(state, player):
    if state.winner is None:
        return 0
    # 1 for winning and -1 for losing
    return 1 if player == state.winner else -1


def paddlePositions(state, action, player):
    enemyPos = state.p2pos if player == 1 else state.p1pos
    myPos = paddleUpdate(state, action, player)

    return myPos, enemyPos
