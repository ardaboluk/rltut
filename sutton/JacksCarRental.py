
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

    def getQ(self,s,a,v):
        """Returns the state-action value for given (s,a) pair.
        That is, Σ(s')[p(s'|s,a) *(r(s,a,s') + y*v(s'))]."""

        if s < 0 or s >= self.numStates:
            raise ValueError("s cannot be less than 0 or greater than {}".format(self.numStates))

        if a < 0 or a >= self.numActions:
            raise ValueError("a cannot be less than 0 or greater than {}".format(self.numActions))

        # decode the number of cars at each location from s
        numLoc1 = s // 21
        numLoc2 = s % 21

        # find the probability of the locations having the queried number of cars at the end of the day.
        # p1 and p2 show the probability of location 1 and 2 having the queried number of cars at the end of the day.
        # meshRow and meshCol corresponds to, respectively, number of cars returned the previous day and requested today.
        # value shows the number of cars at the end of the day if i cars were returned preivous day and j cars were requested today.
        # prob shows the probability of i cars being returned previous day and j cars being requested today.

        # number of cars at the end of the previous day for each location
        values1 = np.tile(np.float(numLoc1),(21,21,21,21))
        values2 = np.tile(np.float(numLoc2),(21,21,21,21))

        # number of cars that are returned the previous day, which become available today
        # this should come before action, because we know the number of cars that will be available
        # before performing the action
        meshRow1 = np.tile(np.array([range(0,21)]).T,(1,21))
        meshRow1 = np.tile(meshRow1,(21,21,1,1))
        meshRow1 = np.rot90(meshRow1.T,k=-1)
        meshRow2 = np.tile(np.array([range(0,21)]).T,(1,21))
        meshRow2 = np.tile(meshRow2,(21,21,1,1))
        values1 = np.clip(values1+meshRow1,0,20)
        values2 = np.clip(values2+meshRow2,0,20)

        minCars = 0
        if a < 5:
            minCars = np.minimum(np.tile(numLoc2,(21,21,21,21))-np.clip(np.tile(numLoc2+(a-5),(21,21,21,21)),0,20),np.clip(values1-(a-5),0,20)-values1)
            values1 += minCars
            values2 -= minCars
        elif a > 5:
            minCars = np.minimum(np.tile(numLoc1,(21,21,21,21))-np.clip(np.tile(numLoc1-(a-5),(21,21,21,21)),0,20),np.clip(values2+(a-5),0,20)-values2)
            values1 -= minCars
            values2 += minCars
                
        # number of cars after rental requests
        meshCol1 = np.tile(np.array([range(0,21)]),(21,1))
        meshCol1 = np.rot90(np.tile(meshCol1,(21,21,1,1)).T)
        meshCol2 = np.rot90(meshCol1,k=-1).T
        values1AfterReq = np.clip(values1-meshCol1,0,20)        
        values2AfterReq = np.clip(values2-meshCol2,0,20)

        # probabilities of location 1 having the values1 numbers of cars
        prob1 = np.multiply((((3**meshRow1)/factorial(meshRow1)) * math.exp(-3)), (((3**meshCol1)/factorial(meshCol1)) * math.exp(-3)))
        # probabilities of location 2 having the values2 numbers of cars
        prob2 = np.multiply((((2**meshRow2)/factorial(meshRow2)) * math.exp(-2)), (((4**meshCol2)/factorial(meshCol2)) * math.exp(-4)))

        prob = np.multiply(prob1, prob2)

        # rewards shows the rewards at each location if #row car returned previous day and #col car rented today
        reqDiff1 = values1 - values1AfterReq
        reqDiff2 = values2 - values2AfterReq
        rewards1 = reqDiff1 * 10
        rewards2 = reqDiff2 * 10
        rewards = rewards1 + rewards2 - minCars*2

        # vectorize each element to get the q-value without loop
        vecValues1AfterReq = np.zeros((21,21,21,21,21,21))
        vecValues2AfterReq = np.zeros((21,21,21,21,21,21))
        vecProb = np.zeros((21,21,21,21,21,21))
        vecRewards = np.zeros((21,21,21,21,21,21))
        vecV = v.reshape(21,21)
        vecNumLoc1prime = np.zeros((21,21,21,21,21,21))
        vecNumLoc2prime = np.zeros((21,21,21,21,21,21))
        
        vecValues1AfterReq[0:21,0:21,:,:,:,:] = values1AfterReq
        vecValues2AfterReq[0:21,0:21,:,:,:,:] = values2AfterReq
        vecProb[0:21,0:21,:,:,:,:] = prob
        vecRewards[0:21,0:21,:,:,:,:] = rewards
        loc2mat, loc1mat = np.meshgrid(np.arange(21),np.arange(21))
        vecNumLoc1prime = np.rot90(np.tile(loc1mat, (21,21,21,21,1,1)).T, k = -1)
        vecNumLoc2prime = np.rot90(np.tile(loc2mat, (21,21,21,21,1,1)).T)

        # logical indexing converted to int8 so that the dimension information isn't lost
        cond = np.logical_and(vecValues1AfterReq == vecNumLoc1prime, vecValues2AfterReq == vecNumLoc2prime).astype(np.int8)
        vecProbCondt = np.multiply(vecProb, cond)
        vecRewardsCondt = np.multiply(vecRewards, cond)
        q = np.sum(np.multiply(vecProbCondt,vecRewardsCondt)) + np.sum(np.multiply(np.sum(vecProbCondt,axis=(2,3,4,5)),  self.gamma * vecV))
        print(q)
        
        return q
