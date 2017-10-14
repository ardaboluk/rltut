
import PolicyIterationViaQ as piq
from JacksCarRental import JacksCarRentalEnvironment as CarEnv
import matplotlib.pyplot as plt
import numpy as np
"""
import PolicyIteration as pi
from GridWorld1 import GridWorldEnvironment as GridEnv

gridEnv = GridEnv(numRows=4, numCols=4, gamma = 0.9)
values, policy = pi.iteratePolicy(gridEnv)
print("Values:\n{}".format(values))
print("Policy:\n{}".format(policy))"""

carenv = CarEnv()
values, policy = piq.iteratePolicy(carenv)
print("Values:\n{}".format(values))
print("Policy:\n{}".format(np.rot90((policy-5).reshape(21,21))))

plt.imshow(np.rot90((policy-5).reshape(21,21)), cmap="hot", interpolation="nearest")
plt.show()
