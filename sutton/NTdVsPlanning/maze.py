
"""Maze module of Example 8.1 in Chapter 8"""

import numpy as np

# number of rows and columns of the grid world
numRows = 6
numCols = 9

# list of walls
wallStates = np.array([7,11,16,20,25,29,41])

# goal state
goalState = 8

def _getNextState(s,a):
    """This function calculates the next state from the given state and action."""

    # get the coordinates from the state
    stateCurRow = s // numCols
    stateCurCol = s % numCols

    # get the horizontal and vertical directions of the action
    actionHorizontal = 0
    actionVertical = 0
    if a == 0:
        actionHorizontal = 1
        actionVertical = 0
    elif a == 1:
        actionHorizontal = 0
        actionVertical = -1
    elif a == 2:
        actionHorizontal = -1
        actionVertical = 0
    elif a == 3:
        actionHorizontal == 0
        actionVertical == 1

    # coordinates of the new state
    stateNextRow = stateCurRow + actionVertical
    stateNextCol = stateCurCol + actionHorizontal

    # the next state
    sprime = stateNextRow * numCols + stateNextCol

    return sprime  


def takeAction(s,a):
    """This function returns the reward and the next state resulting from taking
    action a from state s. The first one is the reward and the next is the state.
    States are indexed from left to right and from top to bottom and the first one is 0.
    The actions 0,1,2,3 are right, up, left and down."""

    reward = 0
    sprime = 0

    # if s is the goal, do nothing. If not, calculate the reward and the next state
    if s == goalState:
        sprime = s
        reward = 0
    else:
        # get sprime
        sprime = _getNextState(s,a)

        # if sprime corresponds to a wall, make it equal to s
        if np.isin(sprime,wallStates):
            sprime = s
        
        # calculate the reward
        if sprime == goalState:
            reward = 1

    return [reward, sprime]
