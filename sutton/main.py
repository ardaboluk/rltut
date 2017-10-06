
import PolicyIteration as pi
from GridWorld1 import GridWorldEnvironment as GridEnv

gridEnv = GridEnv(numRows=4, numCols=4, gamma = 0.9)
pi.iteratePolicy(gridEnv)

"""import PolicyIterationVec as pivec
from JacksCarRental import JacksCarRentalEnvironment as CarEnv

carenv = CarEnv()
pivec.iteratePolicy(carenv)"""
