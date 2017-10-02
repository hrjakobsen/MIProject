import random
random.seed(0)

class Hexagon:
	def __init__(self, x, y):
		self.colour = random.randint(0,4)
		self.x = x
		self.y = y
		self.owner = None

	def __str__(self):
		names = ["o","p"]
		return self.colour.__str__() if self.owner == None else names[self.owner]

	def __repr__(self):
		return self.__str__()

	def setOwner(self, owner):
		self.owner = owner
		return self

class Grid:
	def __init__(self, width, height, obj):
		self.width = width
		self.height = height
		self.initGrid(obj)

	def initGrid(self, obj):
		grid = []
		for h in range(self.height + 1):
			row = []
			for w in range(self.width):
				if h == 0 and w % 2 == 1:
					row.append(None)
				else:
					row.append(obj(w, h))
			grid.append(row)
		self.grid = grid

	def __str__(self):
		return "\n".join(map(str, self.grid))

	def getNeighbours(self, obj):
		x = obj.x
		y = obj.y

		relCoords = [
			(0, -1),
			(1, 0),
			(1, 1),
			(0, 1),
			(-1, 1),
			(-1, 0)
		]

		neighbours = []
		for coord in relCoords:
			newx = x + coord[0]
			newy = y + coord[1]
			if newx < self.width and newx >= 0 and newy <= self.height and newy >= 0 and self.grid[newy][newx] != None:
				neighbours.append(self.grid[newy][newx])
		return neighbours

	def getHexagons(self):
		return [hexagon for row in self.grid for hexagon in filter(lambda x: x != None, row)]

class HexagonGameDriver:
	def __init__(self, player1, player2):
		self.player1 = player1
		self.player2 = player2

		self.board = Grid(9,4, Hexagon)
		self.board.grid[0][0].owner = 0
		self.board.grid[4][8].owner = 1

		self.playGame()

	def playGame(self):
		action = None
		player1Turn = True

		while not self.gameEnded():
			if player1Turn:
				action = self.player1(self.board)
			else: 
				action = self.player2(self.board)
			self.makeMove(player1Turn, action)
			player1Turn = not player1Turn

	def makeMove(self, player1Turn, action):
		player = 0 if player1Turn else 1

		hexagons = filter(lambda x: x.owner == player, self.board.getHexagons())
		while len(hexagons) > 0:
			hexagon = hexagons.pop(0)
			hexagon.colour = action
			neighbours = self.board.getNeighbours(hexagon)

			for neighbour in neighbours:
				if neighbour.owner == None and neighbour.colour == action:
					neighbour.owner = player
					hexagons.append(neighbour)


	def gameEnded(self):
		player1hexagons = filter(lambda x: x.owner == 0, self.board.getHexagons())
		player2hexagons = filter(lambda x: x.owner == 1, self.board.getHexagons())

		neighbours = []
		for hexagon in player1hexagons:
			neighbours += self.board.getNeighbours(hexagon)
		if len(filter(lambda x: x.owner == None, neighbours)) == 0:
			for h in self.board.getHexagons():
				h.owner = 1
			return True

		neighbours = []
		for hexagon in player2hexagons:
			neighbours += self.board.getNeighbours(hexagon)
		if len(filter(lambda x: x.owner == None, neighbours)) == 0:
			for h in self.board.getHexagons():
				h.owner = 0
			return True

		return False

def play1(board):
	print board
	action = int(raw_input("P1: next move? "))
	return action

def play2(board):
	print board
	action = int(raw_input("P2: next move? "))
	return action

game = HexagonGameDriver(play1, play2)