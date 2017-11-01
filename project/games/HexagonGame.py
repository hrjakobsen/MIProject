import numpy as np

def randomGame(width, height):
    """ Generates a random hexagon board
    :param width: width of board
    :param height: height of board at odd columns
    :return: a random game
    """
    board = np.random.randint(5, size=(height + 1, width))

    # What we actually want is a jagged array where the odd cols
    # are shifted half a cell up, to simulate a hexagon grid.
    # We achieve this by making the grid 1 higher than specified and
    # 'remove' the top cell in the odd columns
    for x in range(width // 2):
        board[0][x * 2 + 1] = -1

    # Set the initial player positions
    board[0, 0] = 5
    board[height, width - 1] = 10

    return board


def getHash(board):
    """
    This function generates a unique hash for each board
    This is done by realising that each cell on the board
    can be in one of 7 (5 colors + owned by either player)
    This means that each cell can be stored in a single character
    0-4 (colors), a (owned by player 1) and b (owned by player 2)
    :param board: the board to hash
    :return: hashed string of board
    """

    lookUp = ['0', '1', '2', '3', '4', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', '']
    return ''.join([lookUp[int(i)] for i in np.nditer(board)])


def makeMove(board, neighbourMap, player, action):
    """
    Performs a move on the board and creates an updated version of the board
    :param board: The board to update
    :param player: The player which plays the move
    :param action: The move
    :return: The updated board
    """
    # Do everything on a copy to ensure stateless-ness
    board = board.copy()

    height = board.shape[0]
    width = board.shape[1]

    frontier = getOwnedCells(board, player)

    # Our color can spread through cells that were just added
    # so we maintain a frontier that is the cells that we still
    # need to check for neighbors of the right colour
    while len(frontier):
        point = frontier.pop()
        board[point[0], point[1]] = action + player * 5

        #neighbours = pointsAround(point)
        neighbours = neighbourMap[point[0]][point[1]]

        # Find the neighbours that are inside the board and
        # have the color of the action and add them to the frontier 
        for neighbour in neighbours:
            if (0 <= neighbour[0] < height
                and 0 <= neighbour[1] < width
                and board[neighbour[0], neighbour[1]] != -1
                and board[neighbour[0], neighbour[1]] == action):
                frontier.append(neighbour)

    # If a player has no more moves, the other player is rewarded
    # the rest of the cells on the board
    board = finaliseBoard(board, neighbourMap, player)
    return board


def getOwnedCells(board, player):
    """
    :param board: the board which the cells should be found in
    :param player: the player whose turn currently is
    :return: list of owned cells
    """
    lowerLimit = player * 5
    upperLimit = (player + 1) * 5

    frontier = list(
        np.column_stack(
            np.where(
                np.logical_and(board < upperLimit, board >= lowerLimit)
            )
        )
    )

    return frontier


def finaliseBoard(board, neighbourMap, playerCall):
    """
    Updates the board if the game is terminated, such that players are
    given the remainder of the cells if the opponent cannot reach them.
    :param board: The game to finalise
    :param playerCall: The player calling the function
    :return: The finalised board
    """
    board = board.copy()
    player = 2 if playerCall == 1 else 1
    frontier = getOwnedCells(board, player)
    height = board.shape[0]
    width = board.shape[1]

    playerColor = board[0, 0] if playerCall == 1 else board[height - 1, width - 1]
    found = False

    # Check if the player has a valid action that gains more cells
    for cell in frontier:
        #neighbours = pointsAround(cell)
        neighbours = neighbourMap[cell[0]][cell[1]]
        for neighbour in neighbours:
            if (0 <= neighbour[0] < height
                and 0 <= neighbour[1] < width
                and 0 <= board[neighbour[0], neighbour[1]] < 5):
                found = True
                break
        if found:
            break
    if found:
        return board

    # Award the player the rest of the cells
    for x in np.nditer(board, op_flags=['readwrite']):
        if 0 <= int(x) < 5:
            x[...] = playerColor
    return board


def getReward(game, player):
    """
    This function calculates the reward of a game
    We reward nothing for a move unless it is a winning move
    Then it gains 1 point. If it is a losing move, it gains -1<
    :param game: The current board
    :param player: The current player
    :return: -0.04 for a non-terminal state, 1 for a win, -1 for a loss
    """
    if not gameEnded(game):
        return -0.04

    height = game.shape[0]
    width = game.shape[1]

    player1Color = game[0, 0]
    player2Color = game[height - 1, width - 1]
    player1count = np.count_nonzero(game == player1Color)
    player2count = np.count_nonzero(game == player2Color)

    if player == 1:
        if player1count > player2count:
            return 1
        else:
            return -1
    else:
        if player2count > player1count:
            return 1
        else:
            return -1


def gameEnded(board):
    """
    Checks if the game has ended
    :param board: The current board
    :return: True if the game has ended, False otherwise
    """
    return not np.any(np.logical_and(board >= 0, board < 5))

def pointsAround(point):
    """
    Get the neighbouring coordinates of a point
    :param point: 2-tuple with x and y coordinate
    :return: a list of neighbouring coordinates
    """
    y, x = point[0], point[1]

    relOddCoords = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, -1)
    ]

    relEvenCoords = [
        (-1, 0),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1)
    ]

    odd = x % 2 == 1

    neighbours = []
    for coord in relOddCoords if odd else relEvenCoords:
        newY = y + coord[0]
        newX = x + coord[1]
        neighbours.append([newY, newX])

    return neighbours


def generateNeighbours(width, height):
    neighbourPositions = []

    relOddCoords = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, -1)
    ]

    relEvenCoords = [
        (-1, 0),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1)
    ]

    neighbourPositions = [[0 for x in range(width)] for y in range(height + 1)] 

    for y in range(height + 1):
        for x in range(width):
            odd = x % 2 == 1
            neighbours = []

            for coord in relOddCoords if odd else relEvenCoords:
                newY = y + coord[0]
                newX = x + coord[1]
                neighbours.append([newY, newX])

            neighbourPositions[y][x] = neighbours

    return neighbourPositions


class HexagonGame(object):
    def __init__(self, width, height):
        self.board = randomGame(width, height)
        self.neighbourMap = generateNeighbours(width, height)
        self._hash = getHash(self.board)
        self.width = width
        self.height = height

    def hash(self):
        return self._hash

    def makeMove(self, player, action):
        if action is None:
            return
        self.board = makeMove(self.board, self.neighbourMap, player, action)
        self._hash = getHash(self.board)

    def getReward(self, player):
        return getReward(self.board, player)

    def gameEnded(self):
        return gameEnded(self.board)

    def __deepcopy__(self, _):
        new = HexagonGame(self.width, self.height)
        new.board = self.board
        new.neighbourMap = self.neighbourMap
        new._hash = self._hash
        return new
