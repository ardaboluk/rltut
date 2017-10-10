
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
        values1 = np.tile(numLoc1,(21,21,21))
        values2 = np.tile(numLoc2,(21,21,21))

        # number of cars that are returned the previous day, which become available today
        # this should come before action, because we know the number of cars that will be available
        # before performing the action
        meshRow1 = np.tile(np.array([range(0,21)]).T,(1,21))
        meshRow1 = np.tile(meshRow1,(21,1,1))
        meshRow2 = np.rot90(meshRow1.T, k = -1)
        values1 = np.clip(values1+meshRow1,0,20)
        values2 = np.clip(values2+meshRow2,0,20)

        # number of cars after performing the action (if possible)
        if a < 5 and numLoc2 + (a-5) >= 0:
            cond = values1 - (a-5) <= 20
            values1[cond] -= a-5
            values2[cond] += a-5
        elif a > 5 and numLoc1 - (a-5) >= 0:
            cond = values2 + (a-5) <= 20
            values1[cond] -= a-5
            values2[cond] += a-5

        #values1[cond] = values1[cond]-(a-5)
        #values2[cond] = values2[cond]+(a-5)
            
        # cond1 = np.logical_and(np.tile(numLoc1,(21,21))-(a-5)>=0,values1-(a-5)<=20)
        # cond2 = np.logical_and(np.tile(numLoc2,(21,21))+(a-5)>=0,values2+(a-5)<=20)
        # values1[np.logical_and(cond1,cond2)] = values1[np.logical_and(cond1,cond2)]-(a-5)
        # values2[np.logical_and(cond1,cond2)] = values2[np.logical_and(cond1,cond2)]+(a-5)

        # number of cars after rental requests
        meshCol1 = np.tile(np.array([range(0,21)]),(21,1))
        meshCol1 = np.tile(meshCol1,(21,1,1))
        meshCol2 = meshCol1
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

        # numLoc1primes = np.tile(meshRow.reshape(-1,1),(1,21)).reshape(21,21,21)
        # numLoc2primes = np.tile(numLoc1primes, (21,1,1))
        # numLoc1primes = np.repeat(numLoc1primes, 21, axis = 0)

        # prob1tiled = np.tile(prob1,(441,1,1))
        # prob2tiled = np.tile(prob2,(441,1,1))
        # rewards1tiled = np.tile(rewards1,(441,1,1))
        # rewards2tiled = np.tile(rewards2,(441,1,1))

        # find the total value for performing action a in state s on all sprimes
        # loc1TrueInd = np.where(values1AfterReq == numLoc1primes)
        # loc1TrueInd = np.hstack((np.array(loc1TrueInd[0]).reshape(-1,1), np.array(loc1TrueInd[1]).reshape(-1,1), np.array(loc1TrueInd[2]).reshape(-1,1)))
        # q = 1 + 1
        # q = np.sum(np.multiply(prob1tiled[values1AfterReq == numLoc1primes],rewards1tiled[values1AfterReq == numLoc1primes])) + \
        #          np.sum(np.multiply(prob2tiled[values2AfterReq == numLoc2primes],rewards2tiled[values2AfterReq == numLoc2primes])) + \
        #          np.sum(prob1tiled[values1AfterReq == numLoc1primes]) * \
        #          prob2tiled[np.logical_and(values1AfterReq == numLoc1primes,values2AfterReq == numLoc2primes)]) * self.gamma * v[np.arange(441)]
        # return q

        # find the total value for performing action a in state s on all sprimes
        q = 0
        for sprime in range(0,self.numStates):
            # decode the number of cars at each location from sprime
            numLoc1prime = sprime // 21
            numLoc2prime = sprime % 21

            q += np.sum(np.multiply(prob[values1AfterReq == numLoc1prime],rewards1[values1AfterReq == numLoc1prime])) + \
                 np.sum(np.multiply(prob[values2AfterReq == numLoc2prime],rewards2[values2AfterReq == numLoc2prime])) + \
                 np.sum(prob[np.logical_and(values1AfterReq == numLoc1prime,values2AfterReq == numLoc2prime)]) * self.gamma * v[sprime]

        # the total value is the sum of value of location 1 and value of location 2
        return q
