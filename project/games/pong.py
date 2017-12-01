import numpy as np
import copy
import math

UP = 1
NOTHING = 0
DOWN = -1

pongWidth = 100
pongHeight= 50

class PongGame(object):
    def __init__(self, ballVelocity=None):
        self.width = pongWidth
        self.height = pongHeight
        self.p1pos = self.height // 2
        self.p2pos = self.height // 2
        self.actions = [NOTHING, UP, DOWN]
        if ballVelocity is None:
            direction = np.random.randint(180)
            if direction < 90:
                direction -= 45
            else:
                direction += 45

            dirRad = direction * math.pi / 180
            self.ballVelocity = np.array([math.cos(dirRad), math.sin(dirRad)])

        self.ballPosition = np.array([self.width // 2, self.height // 2])
        self.ballRadius = 2.5
        self.paddleSpeed = 0.5
        self.paddleHeight = 20
        self.winner = None
        self.numFeatures = None

    def __deepcopy__(self, _):
        new = PongGame(self.ballVelocity)
        new.p1pos = self.p1pos
        new.p2pos = self.p2pos
        new.ballPosition = self.ballPosition
        return new

    def getActions(self, player):
        return self.actions

    def gameEnded(self):
        return self.winner is not None

    def getNumFeatures(self):
        if self.numFeatures is None:
            self.numFeatures = len(calculateFeatures(self, 0, 1))

        return self.numFeatures

    def calculateFeatures(self, state, action, player):
        return calculateFeatures(state, action, player)

    def getReward(self, player):
        return getReward(self, player)



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


def getFeaturesOld(player):
    features = [
        lambda s, a: 1,
        # the position of the paddle after a move
        # lambda s, a: paddlePositions(s, a, player)[1],
        # the position of the ball after a move
        # lambda s, a: makePlayerMove(s, a, player).ballPosition[0],
        # lambda s, a: makePlayerMove(s, a, player).ballPosition[1],
        # the velocity of the ball after a move
        # lambda s, a: makePlayerMove(s, a, player).ballVelocity[0],
        # lambda s, a: makePlayerMove(s, a, player).ballVelocity[1],

        lambda s, a: distanceToBall(s, a, player),
    ]

    return features


# one-hot encoding
def getFeatures(player):
    features = [
        lambda s, a: 1,
        lambda s, a: makePlayerMove(s, a, player).ballVelocity[0],
        lambda s, a: makePlayerMove(s, a, player).ballVelocity[1],
        #lambda s, a: distanceToBall(s, a, player)
    ]
    # the height of the board
    for y in range(pongHeight):
        features.append(lambda s, a, yPos=y: oneHotEncodedPaddle(player, yPos, s, a))
    for y in range(pongHeight):
        features.append(lambda s, a, yPos=y: oneHotEncodedBallY(player, yPos, s, a))
    for x in range(pongWidth):
        features.append(lambda s, a, xPos=x: oneHotEncodedBallX(player, xPos, s, a))
    return features


def oneHotEncodedPaddle(player, y, s: PongGame, a):
    newS = makePlayerMove(s, a, player)
    paddlePos = newS.p1pos if player == 1 else newS.p2pos
    return 1 if abs(paddlePos - y) <= newS.paddleHeight // 2 else 0


def oneHotEncodedBallY(player, y, s: PongGame, a):
    newS = makePlayerMove(s, a, player)
    ballPosY = newS.ballPosition[1]
    return 1 if abs(ballPosY - y) <= newS.ballRadius else 0


def oneHotEncodedBallX(player, x, s: PongGame, a):
    newS = makePlayerMove(s, a, player)
    ballPosY = newS.ballPosition[0]
    return 1 if abs(ballPosY - x) <= newS.ballRadius else 0


def distanceToBall(s, a, player):
    s2 = makePlayerMove(s, a, player)
    paddleY = s2.p1pos if player == 1 else s2.p2pos
    return math.log2(1 / max(((s2.ballPosition[1] - paddleY) ** 2), 0.00001))


def calculateFeatures(state, action, player):
    nextState = makePlayerMove(state, action, player)

    results = np.array([
        1,
        #nextState.p1pos if player == 1 else nextState.p2pos,
        #nextState.ballPosition[0],
        #nextState.ballPosition[1],
        #nextState.ballVelocity[0],
        #nextState.ballVelocity[1],
        #distanceToBall(nextState, player),
        getAngle(nextState, player),
        distanceFromCenter(nextState, player)
    ])

    return results

def distanceFromCenter(s, player):
    dist = abs(s.p1pos - s.height // 2) if player == 1 else abs(s.p2pos - s.height // 2)
    return dist

def getAngle(s, player):
    #return 0 if the ball is not travelling in the direction of the player
    #since the agent needs to minimize the angle
    if (player == 1 and s.ballVelocity[0] > 0):
        return 0
    if (player == 2 and s.ballVelocity[0] < 0):
        return 0

    vector = getVectorBetweenBallAndPaddle(s, player)
    lengthOfPaddleVec = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1])
    dotProduct = np.dot(vector, s.ballVelocity)
    angle = math.acos(dotProduct / np.dot(lengthOfPaddleVec, 1))
    angle *= 180/math.pi

    return angle

def getVectorBetweenBallAndPaddle(s, player):
    vector = np.array([0 - s.ballPosition[0], s.p1pos - s.ballPosition[1]]) if player == 1 else np.asarray([200 - s.ballPosition[0], s.p2pos - s.ballPosition[1]])
    return vector

def distanceToBall(s, player):
    paddleY = s.p1pos if player == 1 else s.p2pos

    return (s.ballPosition[1] - paddleY) ** 2

def makePlayerMove(s, a, player):
    return makeMove(s, a, NOTHING) if player == 1 else makeMove(s, NOTHING, a)


def getReward(state: PongGame, player):
    if state.winner is None:
        return 0
    # 1 for winning and -1 for losing
    return 100 if player == state.winner else -100


def paddlePositions(state: PongGame, action, player):
    enemyPos = state.p2pos if player == 1 else state.p1pos
    myPos = paddleUpdate(state, action, player)

    return myPos, enemyPos
