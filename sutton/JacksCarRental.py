
import numpy as np
import math
from scipy.misc import factorial
import os

class JacksCarRentalEnvironment:
    """Environment class for example 4.2 (Jack's Car Rental) in chapter 4.
    There are 441 states such that (0,1);(0,2)...(20,20) where the first number shows the 
    number of cars at the first location and the second number shows the number of cars at 
    the second location. There are 11 actions a0 = 5 car from loc2 to loc1, ..., a5 = 0 car, 
    a6 = 1 car from loc1 to loc2, ..., a10 = 5 car from loc1 to loc2. Due to the stochastic
    nature of the problem, there are no terminal states."""

    def __init__(self):

        self.numStates = 441
        self.numActions = 11
        self.gamma = 0.9
        self.probMat = None
        self.rewardMat = None
        self.terminalStates = []
        
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

        # find the probability of the locations having the queried number of cars at the end of the day
        # p1 and p2 show the probability of location 1 and 2 having the queried number of cars at the end of the day
        # i and j corresponds to, respectively, number of cars returned the previous day and requested today
        # value shows the number of cars at the end of the day if i cars were returned preivous day and j cars were requested today
        # prob shows the probability of i cars being returned previous day and j cars being requested today
        if numLoc1-(a-5)>=0 and numLoc1-(a-5)<=20 and numLoc2+(a-5) >= 0 and numLoc2+(a-5) <= 20:
            values1 = np.tile(numLoc1-(a-5),(21,21))
            values2 = np.tile(numLoc2+(a-5),(21,21))
        else:
            values1 = np.tile(numLoc1,(21,21))
            values2 = np.tile(numLoc2,(21,21))

        meshRow = np.tile(np.array([range(0,21)]).T,(1,21))
        meshCol = np.tile(np.array([range(0,21)]),(21,1))
        values1 = np.clip(values1+meshRow,0,20)
        values1 = np.clip(values1-meshCol,0,20)
        values2 = np.clip(values2+meshRow,0,20)
        values2 = np.clip(values2-meshCol,0,20)
        prob1 = np.multiply((((3**meshRow)/factorial(meshRow)) * math.exp(-3)), (((3**meshCol)/factorial(meshCol)) * math.exp(-3)))
        prob2 = np.multiply((((2**meshRow)/factorial(meshRow)) * math.exp(-2)), (((4**meshCol)/factorial(meshCol)) * math.exp(-4)))
        p1 = np.sum(prob1[values1 == numLoc1prime])
        p2 = np.sum(prob1[values2 == numLoc2prime])

        # the probability of location 1 having numLoc1prime cars AND location 2 having numLoc2prime cars at the and of the day
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

        numLoc1start = 0
        numLoc2start = 0
        
        if numLoc1-(a-5)>=0 and numLoc1-(a-5)<=20 and numLoc2+(a-5) >= 0 and numLoc2+(a-5) <= 20:
            numLoc1start = numLoc1-(a-5)
            numLoc2start = numLoc2+(a-5)
        else:
            numLoc1start = numLoc1
            numLoc2start = numLoc2
        
        # find the reward at the first location
        t = numLoc1start - numLoc1prime
        if t > 0:
            r1 = 10*t

        # find the reward at the second location
        t = numLoc2start - numLoc2prime
        if t > 0:
            r2 = 10*t

        # total reward
        return r1 + r2        

    def getEnvironmentMatrices(self):
        """If the matrices for transition probabilities and rewards exist as files, load them back.
        If not, populates the transition probabilities matrix and rewards matrix by calling 
        getProbability and getReward methods of the environment on every possible 
	s-a-sprime method. Then, dumps them to files. Saving matrices beforehand is useful 
	for vectorized policy iteration."""

        # check if the transition probability matrix exist, if so load it back, if not create and save it

        if os.path.isfile("probMat") :
            print("Loading the probability matrix from file.")
            self.probMat = np.load("probMat")
        else:
            print("Creating the probability matrix.")
            self.probMat = np.zeros((self.numStates, self.numActions, self.numStates))
            
            for s in range(0,self.numStates):
                print("Populating the probability matrix for state {}".format(s))
                for a in range(0,self.numActions):
                    for sprime in range(0,self.numStates):
                        self.probMat[s,a,sprime] = self.getProbability(s,a,sprime)
		        
            self.probMat.dump("probMat")
        
        if os.path.isfile("rewardMat"):
            print("Loading reward matrix from file.")
            self.rewardMat = np.load("rewardMat")
        else:
            print("Creating the reward matrix.")
            self.rewardMat = np.zeros((self.numStates, self.numActions, self.numStates))
            
            for s in range(0,self.numStates):
                print("Populating the reward matrix for state {}".format(s))
                for a in range(0,self.numActions):
                    for sprime in range(0,self.numStates):
                        self.rewardMat[s,a,sprime] = self.getReward(s,a,sprime)
		        
            self.rewardMat.dump("rewardMat")
