
import numpy as np
import tensorflow as tf
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

        # mesh matrices for returned and requested cars for two locations
        # self.__meshRow1 = np.tile(np.array([range(0,21)]).T,(1,21))
        # self.__meshRow1 = np.tile(self.__meshRow1,(21,21,1,1))
        # self.__meshRow1 = np.rot90(self.__meshRow1.T,k=-1)
        # self.__meshRow2 = np.tile(np.array([range(0,21)]).T,(1,21))
        # self.__meshRow2 = np.tile(self.__meshRow2,(21,21,1,1))

        # self.__meshCol1 = np.tile(np.array([range(0,21)]),(21,1))
        # self.__meshCol1 = np.rot90(np.tile(self.__meshCol1,(21,21,1,1)).T)
        # self.__meshCol2 = np.rot90(self.__meshCol1,k=-1).T
        
        self.__meshRow1 = np.tile(np.array([range(0,21)]).T,(1,21))
        self.__meshRow1 = np.tile(self.__meshRow1,(21,21,1,1))
        self.__meshRow1 = np.rot90(self.__meshRow1.T,k=-1)
        self.__meshRow2 = np.tile(np.array([range(0,21)]).T,(1,21))
        self.__meshRow2 = np.tile(self.__meshRow2,(21,21,1,1))
        
        self.__meshCol1 = np.tile(np.array([range(0,21)]),(21,1))
        self.__meshCol1 = np.rot90(np.tile(self.__meshCol1,(21,21,1,1)).T)
        self.__meshCol2 = np.rot90(self.__meshCol1,k=-1).T

        # probabilities of location 1 having the values1 numbers of cars
        prob1 = np.multiply((((3**self.__meshRow1)/factorial(self.__meshRow1)) * math.exp(-3)), (((3**self.__meshCol1)/factorial(self.__meshCol1)) * math.exp(-3)))
        # probabilities of location 2 having the values2 numbers of cars
        prob2 = np.multiply((((2**self.__meshRow2)/factorial(self.__meshRow2)) * math.exp(-2)), (((4**self.__meshCol2)/factorial(self.__meshCol2)) * math.exp(-4)))
        self.__prob = np.multiply(prob1, prob2)

        # construct tensorflow computation graph
        # self.__sess = tf.Session()
        # self.__init = tf.global_variables_initializer()
        # self.__vecValues1AfterReqPlc = tf.placeholder(tf.int8, [21,21,21,21,21,21])
        # self.__vecValues2AfterReqPlc = tf.placeholder(tf.int8, [21,21,21,21,21,21])
        # self.__vecNumLoc1primePlc = tf.placeholder(tf.int8, [21,21,21,21,21,21])
        # self.__vecNumLoc2primePlc = tf.placeholder(tf.int8, [21,21,21,21,21,21])
        # self.__vecProbPlc = tf.placeholder(tf.float32, [21,21,21,21,21,21])
        # self.__vecRewardsPlc = tf.placeholder(tf.float32, [21,21,21,21,21,21])
        # self.__vecVPlc = tf.placeholder(tf.float32, [21,21])

        # cond1 = tf.equal(self.__vecValues1AfterReqPlc, self.__vecNumLoc1primePlc)
        # cond2 = tf.equal(self.__vecValues2AfterReqPlc, self.__vecNumLoc2primePlc)
        # condNum = tf.cast(tf.logical_and(cond1,cond2), tf.float32)
        # vecProbCondt = tf.multiply(self.__vecProbPlc, condNum)
        # vecRewardsCondt = tf.multiply(self.__vecRewardsPlc, condNum)
        # self.__q = tf.reduce_sum(tf.multiply(vecProbCondt,vecRewardsCondt)) + tf.reduce_sum(tf.multiply(tf.reduce_sum(vecProbCondt,axis=(2,3,4,5)),  tf.multiply(self.gamma, self.__vecVPlc)))
        
        self.__sess = tf.Session()
        self.__init = tf.global_variables_initializer()
        self.__values1AfterReqPlc = tf.placeholder(tf.int8, [21,21,21,21])
        self.__values2AfterReqPlc = tf.placeholder(tf.int8, [21,21,21,21])
        self.__numLoc1primePlc = tf.placeholder(tf.int8, shape = ())
        self.__numLoc2primePlc = tf.placeholder(tf.int8, shape = ())
        self.__probPlc = tf.placeholder(tf.float32, [21,21,21,21])
        self.__rewardsPlc = tf.placeholder(tf.float32, [21,21,21,21])
        self.__vPlc = tf.placeholder(tf.float32, shape = ())

        cond1 = tf.equal(self.__values1AfterReqPlc, self.__numLoc1primePlc)
        cond2 = tf.equal(self.__values2AfterReqPlc, self.__numLoc2primePlc)
        cond = tf.logical_and(cond1,cond2)
        probCondt = tf.boolean_mask(self.__probPlc, cond)
        rewardsCondt= tf.boolean_mask(self.__rewardsPlc, cond)
        self.__qsingle = tf.reduce_sum(tf.multiply(probCondt, rewardsCondt)) + tf.multiply(tf.reduce_sum(probCondt),  tf.multiply(self.__vPlc, self.gamma))
        

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
        values1 = np.clip(values1+self.__meshRow1,0,20)
        values2 = np.clip(values2+self.__meshRow2,0,20)

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
        values1AfterReq = np.clip(values1-self.__meshCol1,0,20)        
        values2AfterReq = np.clip(values2-self.__meshCol2,0,20)

        # rewards shows the rewards at each location if #row car returned previous day and #col car rented today
        reqDiff1 = values1 - values1AfterReq
        reqDiff2 = values2 - values2AfterReq
        rewards1 = reqDiff1 * 10
        rewards2 = reqDiff2 * 10
        rewards = rewards1 + rewards2 - minCars*2

        # vectorize each element to get the q-value without loop
        # vecValues1AfterReq = np.zeros((21,21,21,21,21,21))
        # vecValues2AfterReq = np.zeros((21,21,21,21,21,21))
        # vecProb = np.zeros((21,21,21,21,21,21))
        # vecRewards = np.zeros((21,21,21,21,21,21))
        # vecV = v.reshape(21,21)
        # vecNumLoc1prime = np.zeros((21,21,21,21,21,21))
        # vecNumLoc2prime = np.zeros((21,21,21,21,21,21))
        
        # vecValues1AfterReq[0:21,0:21,:,:,:,:] = values1AfterReq
        # vecValues2AfterReq[0:21,0:21,:,:,:,:] = values2AfterReq
        # vecProb[0:21,0:21,:,:,:,:] = self.__prob
        # vecRewards[0:21,0:21,:,:,:,:] = rewards
        # loc2mat, loc1mat = np.meshgrid(np.arange(21),np.arange(21))
        # vecNumLoc1prime = np.rot90(np.tile(loc1mat, (21,21,21,21,1,1)).T, k = -1)
        # vecNumLoc2prime = np.rot90(np.tile(loc2mat, (21,21,21,21,1,1)).T)

        # logical indexing converted to int8 so that the dimension information isn't lost
        # cond = np.logical_and(vecValues1AfterReq == vecNumLoc1prime, vecValues2AfterReq == vecNumLoc2prime).astype(np.int8)
        # vecProbCondt = np.multiply(vecProb, cond)
        # vecRewardsCondt = np.multiply(vecRewards, cond)
        # q = np.sum(np.multiply(vecProbCondt,vecRewardsCondt)) + np.sum(np.multiply(np.sum(vecProbCondt,axis=(2,3,4,5)),  self.gamma * vecV))

        self.__sess.run(self.__init)
        
        qSum = 0
        for sprime in range(self.numStates):
        
            numLoc1prime = sprime // 21
            numLoc2prime = sprime % 21
        
            tf_feed_dict = {self.__values1AfterReqPlc : values1AfterReq,
                            self.__values2AfterReqPlc : values2AfterReq,
                            self.__numLoc1primePlc : numLoc1prime,
                            self.__numLoc2primePlc : numLoc2prime,
                            self.__probPlc : self.__prob,
                            self.__rewardsPlc : rewards,
                            self.__vPlc : v[sprime]}
                            
            qSum += self.__sess.run(self.__qsingle, feed_dict=tf_feed_dict)
			
        return qSum
