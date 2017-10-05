

import numpy as np

class GridWorldEnvironment:
    """Environment for example 4.1 (deterministic grid world) in chapter 4.
    4x4 grid world. 15 states where (0,0) and (15,15) are considered one terminal state.
    Non-terminal states are 1...14 from left to right. Actions are 0 right 1 up 2 left 3 down.
    """

    def __init__(self):

        self.numStates = 16
        self.numActions = 4
        self.gamma = 0.9

    def getProbability(self,s,a,sprime):

        if s < 0 or s >= self.numStates:
            raise ValueError("s cannot be less than 0 or greater than {}".format(self.numStates))

        if sprime < 0 or sprime >= self.numStates:
            raise ValueError("sprime cannot be less than 0 or greater than {}".format(self.numStates))

        if a < 0 or a >= self.numActions:
            raise ValueError("a cannot be less than 0 or greater than {}".format(self.numActions))
        # if terminal states are reached, return 0
        if s == 0 or s == 15:
            return 0

        # position indicators
        pos_h = 0
        pos_v = 0
        new_pos_h = 0
        new_pos_v = 0

        # decode position from s
        pos_h = s // 4
        pos_v = s % 4

        # decode new position from sprime
        new_pos_h = sprime // 4
        new_pos_v = sprime % 4

        # valid state-action pairs which move the agent
        is_legal = a == 0 and (pos_v == new_pos_v - 1) and (pos_h == new_pos_h)
        is_legal = is_legal or (a == 1 and (pos_h == new_pos_h + 1) and (pos_v == new_pos_v))
        is_legal = is_legal or (a == 2 and (pos_v == new_pos_v + 1) and (pos_h == new_pos_h))
        is_legal = is_legal or (a == 3 and (pos_h == new_pos_h - 1) and (pos_v == new_pos_v))
        
        # valid state-action pairs which don't move the agent
        is_legal = is_legal or (a == 0 and pos_v == 3 and new_pos_v == 3 and pos_h == new_pos_h)
        is_legal = is_legal or (a == 1 and pos_h == 0 and new_pos_h == 0 and pos_v == new_pos_v)
        is_legal = is_legal or (a == 2 and pos_v == 0 and new_pos_v == 0 and pos_h == new_pos_h)
        is_legal = is_legal or (a == 3 and pos_h == 3 and new_pos_h == 3 and pos_v == new_pos_v)

        return np.float(is_legal)

    def getReward(self,s,a,sprime):

        if sprime == 0 or sprime == 15:
            return 0
        else:
            return -1
