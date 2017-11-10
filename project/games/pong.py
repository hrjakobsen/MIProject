import numpy as np
import copy
import math

UP = 1
NOTHING = 0
DOWN = -1

class PongGame(object):
    def __init__(self):
        self.width = 500
        self.height = 200
        self.p1pos = 10
        self.p2pos = self.height // 2
        self.actions = [UP, DOWN, NOTHING]
        self.ballVelocity = np.array([-2, -0.000000000000001])
        self.ballPosition = np.array([10, 10])
        self.ballRadius = 2.5
        self.paddleSpeed = 3
        self.paddleHeight = 20




def updateBall(state: PongGame):
    newPos = state.ballPosition + state.ballVelocity
    newVel = state.ballVelocity
    movingLeft = state.ballVelocity[0] < 0

    factorPaddle, factorWall = factorToPaddle(state), factorToWall(state)

    if factorPaddle > 1 and factorWall > 1:
        return newPos, state.ballVelocity, None

    if factorWall < 1 and factorPaddle < 1:
        #Update the smallest factor first
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
                return None, None, int(paddlePos < state.width // 2) + 1
    elif factorWall < 1:
        newPos = state.ballPosition + state.ballVelocity * factorWall
        newVel = np.dot(state.ballVelocity, np.array([1, -1]))
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
            return None, None, int(paddlePos < state.width // 2) + 1

    return newPos, newVel, None

def factorToWall(state: PongGame):
    """
    this function returns a factor that multiplied to the velocity vector results in
    hitting the top or bottom of the board depending on direction
    :param state:
    :return:
    """
    return max([(wall - state.ballPosition[1])/state.ballVelocity[1] for wall in [state.ballRadius, state.height - state.ballRadius]])

def factorToPaddle(state: PongGame):
    """
    this function returns a factor that multiplied to the velocity vector results in
    hitting the left side or right side of the board depending on direction
    :param state:
    :return:
    """
    return max([(paddle - state.ballPosition[0]) / state.ballVelocity[0] for paddle in [0, state.width]])


def makeMove(state: PongGame, action1, action2):
    newState = copy.deepcopy(state)

    newState.p1pos = paddleUpdate(state, action1, 1)
    newState.p2pos = paddleUpdate(state, action2, 2)

    newState.ballPosition, newState.ballVelocity = updateBall(state)

def paddleUpdate(state, action, player):
    myPos = state.p1pos if player == 1 else state.p2pos
    return max(min(state.height - state.paddleHeight // 2, myPos + state.paddleSpeed * action), state.paddleHeight // 2)


def getFeatures(player):

    features = [
        lambda s, a: paddlePositions(s, a, player)[0],
        lambda s, a: paddlePositions(s, a, player)[1],
    ]

    return features


def paddlePositions(state: PongGame, action, player):
    enemyPos = state.p2pos if player == 1 else state.p1pos
    myPos = paddleUpdate(state, action, player)

    return myPos, enemyPos
