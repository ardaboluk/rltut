

import numpy as np

class GridWorldEnvironment:
    """Environment for example 4.1 (deterministic grid world) in chapter 4.
    4x4 grid world. 15 states where (0,0) and (15,15) are considered one terminal state.
    Non-terminal states are 1...14 from left to right. Actions are 0 right 1 up 2 left 3 down.
    """

    def __init__(self):

        self.gamma = 1
        self.numStates = 15
        self.numActions = 4

    def getProbability(s,a,sprime):

        if s < 0 || s >= self.numStates:
            raise ValueError("s cannot be less than 0 or greater than {}".format(self.numStates))

        if sprime < 0 || sprime >= self.numStates:
            raise ValueError("sprime cannot be less than 0 or greater than {}".format(self.numStates))

        if a < 0 || a >= self.numActions:
            raise ValueError("a cannot be less than 0 or greater than {}".format(self.numActions))

        # position indicators
        pos_h = 0
        pos_v = 0
        new_pos_h = 0
        new_pos_v = 0
        
        # if terminal states are reached, return 0
        if s == 0 || s == 15:
            return 0

        # decode position from s
        pos_h = s // 4
        pos_v = s % 4

        # decode new position from sprime
        new_pos_h = sprime // 4
        new_pos_v = sprime % 4
        
        if a == 0 && pos_h += new_pos_h + 1:
            



