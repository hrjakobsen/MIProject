import random
random.seed(0)

class Hexagon:
	def __init__(self, x, y):
		self.colour = random.randint(0,4)
		self.x = x
		self.y = y
		self.owner = None

	def __str__(self):
		#the names of each player to show which hexagons the player owns
		names = ["o","p"]
		#colour is a number
		return (self.colour.__str__() if self.owner == None else names[self.owner])

	def __repr__(self):
		return self.__str__()

	def setOwner(self, owner):
		self.owner = owner
		return self

class Grid:
	#we define the grid
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

	#which neighbours each hexagon has
	def getNeighbours(self, obj):
		x = obj.x
		y = obj.y
		#the coordinates the neighbours can have from the chosen hexagon
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

		neighbours = []
		for coord in relEvenCoords if x % 2 == 0 else relOddCoords:
			newy = y + coord[0]
			newx = x + coord[1]
			if newx < self.width and newx >= 0 and newy <= self.height and newy >= 0 and self.grid[newy][newx] != None:
				neighbours.append(self.grid[newy][newx])
		return neighbours

	def getHexagons(self):
		return [hexagon for row in self.grid for hexagon in filter(lambda x: x != None, row)]

class HexagonGameDriver:
	def __init__(self, player1, player2, finish=None):
		self.player1 = player1
		self.player2 = player2
		#size of the grid
		self.board = Grid(9,4, Hexagon)
		#where each of the players start
		self.board.grid[0][0].owner = 0
		self.board.grid[4][8].owner = 1
		#callback for finished game
		self.finish = finish
		#the game starts
		self.playGame()

	def playGame(self):
		action = None
		player1Turn = True

		#while the game is not over
		while not self.gameEnded():
			if player1Turn:
				action = self.player1(self.board)
			else: 
				action = self.player2(self.board)
			self.makeMove(player1Turn, action)
			player1Turn = not player1Turn
		if self.finish != None:
			self.finish(self.board)

	def makeMove(self, player1Turn, action):
		player = 0 if player1Turn else 1

		hexagons = filter(lambda x: x.owner == player, self.board.getHexagons())
		while len(hexagons) > 0:
			hexagon = hexagons.pop(0) #get the latest hexagon
			hexagon.colour = action
			neighbours = self.board.getNeighbours(hexagon)

			#we add the new neighbours to the hexagons pr player
			for neighbour in neighbours:
				if neighbour.owner == None and neighbour.colour == action:
					neighbour.owner = player
					hexagons.append(neighbour)

	#to find out when the game is over
	def gameEnded(self):
		player1hexagons = filter(lambda x: x.owner == 0, self.board.getHexagons())
		player2hexagons = filter(lambda x: x.owner == 1, self.board.getHexagons())

		neighbours = []
		for hexagon in player1hexagons: #if the player has more moves available
			neighbours += self.board.getNeighbours(hexagon)
		if len(filter(lambda x: x.owner == None, neighbours)) == 0:
			for h in filter(lambda x: x.owner == None, self.board.getHexagons()):
				h.owner = 1
			return True

		neighbours = []
		for hexagon in player2hexagons:
			neighbours += self.board.getNeighbours(hexagon)
		if len(filter(lambda x: x.owner == None, neighbours)) == 0:
			for h in filter(lambda x: x.owner == None, self.board.getHexagons()):
				h.owner = 0
			return True

		return False

#the board for player 1
def play1(board):
	print(board)
	action = int(raw_input("P1: next move? "))	
	return action

#the board for player 2
def play2(board):
	print(board)
	action = int(raw_input("P2: next move? "))
	return action