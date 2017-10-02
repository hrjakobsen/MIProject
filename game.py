import hexagongame
import math
from Tkinter import *
import threading
import time

def getPointsForPoly(startX, startY):
    r = 50
    n = 6
    points = []
    for i in range(6):
        points += [startX + r * math.cos(2 * math.pi * i / n), startY + r * math.sin(2 * math.pi * i / n)]
    return points

def select0():
    global playerAction
    playerAction = 0

def select1():
    global playerAction
    playerAction = 1

def select2():
    global playerAction
    playerAction = 2

def select3():
    global playerAction
    playerAction = 3

def select4():
    global playerAction
    playerAction = 4

root = Tk()

f = Frame(root, height=32, width=200)
f.grid(row=2, column=0, columnspan=2)  
f.pack_propagate(0) # don't shrink
f.pack()

Button(f, text="", width=10, bg="red", activebackground="red", command=select0).grid(row=0, column=0)
Button(f, text="", width=10, bg="green", activebackground="green", command=select1).grid(row=0, column=1)
Button(f, text="", width=10, bg="blue", activebackground="blue", command=select2).grid(row=0, column=2)
Button(f, text="", width=10, bg="gray", activebackground="gray", command=select3).grid(row=0, column=3)
Button(f, text="", width=10, bg="yellow", activebackground="yellow", command=select4).grid(row=0, column=4)


playerAction = None

lastFrame = None

def drawBoard(board):
    global lastFrame

    colors = ["red", "green", "blue", "gray", "yellow"]
    owner = ["o", "p"]
    if lastFrame != None:
        lastFrame.destroy()
    lastFrame = Canvas(root, width=900, height=600)
    
    for hexagon in board.getHexagons():
        canvas = lastFrame

        canvas.create_polygon(getPointsForPoly(53 + hexagon.x * 100, (52 if hexagon.x % 2 == 0 else 2) + hexagon.y * 100), 
                       outline= "black", 
                       fill= "black", 
                       width=0)

        canvas.create_polygon(getPointsForPoly(50 + hexagon.x * 100, (50 if hexagon.x % 2 == 0 else 0) + hexagon.y * 100), 
                       outline= colors[hexagon.colour], 
                       fill= colors[hexagon.colour], 
                       width=0)
        if hexagon.owner != None:
            canvas.create_text([50 + hexagon.x * 100, (50 if hexagon.x % 2 == 0 else 0) + hexagon.y * 100], text=owner[hexagon.owner])
    lastFrame.pack()

def player(board):
    print board
    global playerAction
    drawBoard(board)
    while playerAction == None:
        time.sleep(.5)
    action = playerAction
    playerAction = None
    return action



def runGame():
    game = hexagongame.HexagonGameDriver(player, player, finish=drawBoard)
    
t = threading.Thread(target=runGame)
t.start()



root.mainloop()

