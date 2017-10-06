
import numpy as np
import math
import os

class JacksCarRentalEnvironment:
    """Environment class for example 4.2 (Jack's Car Rental) in chapter 4.
    There are 441 states such that (0,1);(0,2)...(20,20) where the first number shows the 
    number of cars at the first location and the second number shows the number of cars at 
    the second location. There are 11 actions a0 = 5 car from loc2 to loc1, ..., a5 = 0 car, 
    a6 = 1 car from loc1 to loc2, ..., a10 = 5 car from loc1 to loc2."""

    def __init__(self):

        self.numStates = 441
        self.numActions = 11
        self.gamma = 0.9
        self.probMat = None
        self.rewardMat = None

    def getProbability(self,s,a,sprime):

        if s < 0 or s >= self.numStates:
            raise ValueError("s cannot be less than 0 or greater than {}".format(self.numStates))

        if sprime < 0 or sprime >= self.numStates:
            raise ValueError("sprime cannot be less than 0 or greater than {}".format(self.numStates))

        if a < 0 or a >= self.numActions:
            raise ValueError("a cannot be less than 0 or greater than {}".format(self.numActions))

        # decode the number of cars at each location from s
        numLoc1 = s // 21
        numLoc2 = s % 21

        # decode the number of cars at each location from sprime
        numLoc1prime = sprime // 21
        numLoc2prime = sprime % 21

        # check obvious situations where the probability is 0
        if numLoc1-(a-5) < 0 or numLoc1-(a-5) > 20 or numLoc2+(a-5) < 0 or numLoc2+(a+5) > 20:
            return 0

        # find the probability of the location 1 having the queried number of cars at the end of the day
        # p1 shows the probability of location 1 having the queried number of cars at the end of the day
        # i and j corresponds to, respectively, number of cars returned the previous day and requested today
        # value shows the number of cars at the end of the day if i cars were returned preivous day and j cars were requested today
        # prob shows the probability of i cars being returned previous day and j cars being requested today
        p1 = 0
        for i in range(0,21):
            for j in range(0,21):
                value = (i-j) + (numLoc1-(a-5))
                value = value if (value >= 0) else 0
                value = value if (value <= 20 ) else 20
                prob = (((3**i)/math.factorial(i)) * math.exp(-3)) * (((3**j)/math.factorial(j)) * math.exp(-3))

                if value == numLoc1prime:
                    p1 += prob

        # similarly, find the probability of the location 2 having the queried number of cars at the end of the day
        p2 = 0
        for i in range(0,21):
            for j in range(0,21):
                value = (i-j) + (numLoc2+(a-5))
                value = value if (value >= 0) else 0
                value = value if (value <= 20 ) else 20
                prob = (((2**i)/math.factorial(i)) * math.exp(-2)) * (((4**j)/math.factorial(j)) * math.exp(-4))

                if value == numLoc1prime:
                    p2 += prob

        # the probability of location 1 having numLoc1prime cars and location 2 having numLoc2prime cars at the and of the day
        return p1 * p2
    

    def getReward(self,s,a,sprime):

        # decode the number of cars at each location from s
        numLoc1 = s // 21
        numLoc2 = s % 21

        # decode the number of cars at each location from sprime
        numLoc1prime = sprime // 21
        numLoc2prime = sprime % 21

        # rewards from the location 1 and location 2
        r1 = 0
        r2 = 0
        
        # find the reward at the first location
        t = (numLoc1 - (a-5)) - numLoc1prime
        if t > 0:
            r1 = 10*t

        # find the reward at the second location
        t = (numLoc2 + (a-5)) - numLoc2prime
        if t > 0:
            r2 = 10*t

        # total reward
        return r1 + r2
    
    def saveEnvironmentMatrices(self):
        """Populates the transition probabilities matrix and rewards matrix by calling 
        getProbability and getReward methods of the environment on every possible 
	s-a-sprime method. Then, dumps them to files. Saving matrices beforehand is useful 
	for vectorized policy iteration."""
	
        self.probMat = np.zeros((self.numStates, self.numActions, self.numStates))
        self.rewardMat = np.zeros((self.numStates, self.numActions, self.numStates))
        
        for s in range(0,self.numStates):
            print("Populating matrices for state {}".format(s))
            for a in range(0,self.numActions):
                for sprime in range(0,self.numStates):
                    self.probMat[s,a,sprime] = self.getProbability(s,a,sprime)
                    self.rewardMat[s,a,sprime] = self.getReward(s,a,sprime)
		    
        self.probMat.dump("probMat")
        self.rewardMat.dump("rewardMat")

    def loadEnvironmentMatrices(self):
        """Loads the transition probabilities and rewards matrices from files.
        Shape of the matrices are (numStates, numActions, numStates)."""

        # check if the files exist, if so load them back and return
        if os.path.isfile("probMat") and os.path.isfile("rewardMat"):
            self.probMat = np.load("probMat")
            self.rewardMat = np.load("rewardMat")
        else:
            print("Please create files for probabilitiy and reward matrices first.")
            sys.exit()


