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
        self.p1pos = self.height // 2
        self.p2pos = self.height // 2
        self.actions = [UP, DOWN, NOTHING]
        direction = np.random.randint(180, size=1)[0]
        if direction < 90:
            direction -= 45
        else:
            direction += 45

        velocity = [math.cos(direction * math.pi / 180), math.sin(direction * math.pi / 180)]
        self.ballVelocity = np.array(velocity)
        self.ballPosition = np.array([self.width // 2, self.height // 2])
        self.ballRadius = 2.5
        self.paddleSpeed = 0.5
        self.paddleHeight = 20
        self.winner = None

    def getActions(self):
        return self.actions

    def gameEnded(self):
        return self.winner is not None

    def getFeatures(self, player):
        return getFeatures(player)

    def getReward(self, player):
        return getReward(self, player)

    def __deepcopy__(self, _):
        new = PongGame()
        new.p1pos = self.p1pos
        new.p2pos = self.p2pos
        new.ballVelocity = self.ballVelocity
        new.ballPosition = self.ballPosition
        return new


def updateBall(state: PongGame):
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


def factorToWall(state: PongGame):
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

    newState.ballPosition, newState.ballVelocity, newState.winner = updateBall(state)

    return newState


def paddleUpdate(state, action, player):
    myPos = state.p1pos if player == 1 else state.p2pos
    return max(min(state.height - state.paddleHeight // 2, myPos + state.paddleSpeed * action), state.paddleHeight // 2)


def getFeatures(player):
    features = [
        lambda s, a: 1,
        # the position of the paddle after a move
        # lambda s, a: paddlePositions(s, a, player)[1],
        # the position of the ball after a move
        # lambda s, a: makePlayerMove(s, a, player).ballPosition[0],
        # lambda s, a: makePlayerMove(s, a, player).ballPosition[1],
        # the velocity of the ball after a move
        # lambda s, a: makePlayerMove(s, a, player).ballVelocity[0],
        # ambda s, a: makePlayerMove(s, a, player).ballVelocity[1],

        lambda s, a: distanceToBall(s, a, player),
    ]

    return features


def distanceToBall(s, a, player):
    s2 = makePlayerMove(s, a, player)
    paddleY = s2.p1pos if player == 1 else s2.p2pos
    return math.log2(1 / max(((s2.ballPosition[1] - paddleY) ** 2), 0.00001))


def makePlayerMove(s, a, player):
    return makeMove(s, a, NOTHING) if player == 1 else makeMove(s, NOTHING, a)


def getReward(state: PongGame, player):
    if state.winner is None:
        return 0
    # 1 for winning and -1 for losing
    return 1 if player == state.winner else -1


def paddlePositions(state: PongGame, action, player):
    enemyPos = state.p2pos if player == 1 else state.p1pos
    myPos = paddleUpdate(state, action, player)

    return myPos, enemyPos
