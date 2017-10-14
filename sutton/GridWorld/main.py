
import matplotlib.pyplot as plt
import numpy as np
import PolicyIteration as pi
from GridWorld1 import GridWorldEnvironment as GridEnv

gridEnv = GridEnv(numRows=4, numCols=4, gamma = 0.9)
values, policy = pi.iteratePolicy(gridEnv)
print("Values:\n{}".format(values))
print("Policy:\n{}".format(policy))

plt.imshow(policy.reshape(4,4), cmap="hot", interpolation="nearest")
plt.show()
