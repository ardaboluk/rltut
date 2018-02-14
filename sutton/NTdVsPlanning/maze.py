
import numpy as np
import matplotlib.pyplot as plt

class Maze:
    """General class for grid world. Done for the maze in Example 8.1 in Chapter 8"""

    def __init__(self, numRows, numCols, wallStates, goalStates):

        self.numRows = numRows
        self.numCols = numCols
        self.wallStates = wallStates
        self.goalStates = goalStates

    def __getNextState(self, s, a):
        """This function calculates the next state from the given state and action."""

        # get the coordinates from the state
        stateCurRow = s // self.numCols
        stateCurCol = s % self.numCols

        # get the horizontal and vertical directions of the action
        # 0,1,2,3 corresponds to right,up,left,down
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
            actionHorizontal = 0
            actionVertical = 1

        # coordinates of the new state
        stateNextRow = stateCurRow + actionVertical
        stateNextCol = stateCurCol + actionHorizontal

        if stateNextRow < 0 or stateNextRow >= self.numRows:
            stateNextRow = stateCurRow
        
        if stateNextCol < 0 or stateNextCol >= self.numCols:
            stateNextCol = stateCurCol

        # the next state
        sprime = stateNextRow * self.numCols + stateNextCol

        # if sprime corresponds to a wall, make it equal to s
        if np.isin(sprime, self.wallStates):
            sprime = s

        return sprime  
    
    def takeAction(self, s, a):
        """This function returns the reward and the next state resulting from taking
        action a from state s. The first one is the reward and the next one is the next state.
        States are indexed from left to right and from top to bottom and the first one is 0.
        The actions 0,1,2,3 are right, up, left and down."""

        reward = 0
        sprime = 0

        # if s is the goal, do nothing. If not, calculate the reward and the next state
        if np.isin(s, self.goalStates):
            sprime = s
            reward = 0
        else:
            # get sprime
            sprime = self.__getNextState(s,a)
            
            # calculate the reward
            if np.isin(sprime, self.goalStates):
                reward = 1

        return [reward, sprime]

    def drawPolicy(self, policy):
        """Utility method for drawing a given policy. Policy is drawn on a grid of [numRows, numCols]."""

        for i in range(self.numRows):
            for j in range(self.numCols):

                marker = ''
                color = ''

                state_num = i * self.numCols + j
                if np.isin(state_num, self.wallStates):
                    marker = 's'
                    color = 'g'
                elif np.isin(state_num, self.goalStates):
                    marker = 'h'
                    color = 'r'
                else:               

                    max_action_ind = np.argmax(policy[i * self.numCols + j])

                    if max_action_ind == 0:
                        marker = '>'
                    elif max_action_ind == 1:
                        marker = '^'
                    elif max_action_ind == 2:
                        marker = '<'
                    elif max_action_ind == 3:
                        marker = 'v'
                    
                    color = 'k'

                plt.plot([j], [self.numRows - i - 1], "{}{}".format(marker, color), markersize=10)

        plt.axis([-1, self.numCols, -1, self.numRows])
        plt.show() 
