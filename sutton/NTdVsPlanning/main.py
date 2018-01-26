
import numpy as np
from maze import Maze
from nsarsa import NSarsa

grid = Maze(6,9,np.array([7,11,16,20,25,29,41]), np.array([8]))

ntd = NSarsa(54,4,[8],grid)

ntd.estimateQ(0.1,0.95,0.1,4,10000)

print(ntd.getQValues())

print()

print(ntd.getPolicy())

grid.drawPolicy(ntd.getPolicy())