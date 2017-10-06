
"""
Vectorized policy iteration (for Chapter 4 Dynamic Programming).
env shold have the following methods and fields:
numStates: the number of states.
numActions: the number of actions.
gamma: discount factor for the environment.
probMat: Matrix of transition probabilities. Shape (numStates, numActions, numStates).
rewardMat: Matrix of rewards. Shape (numStates, numActions, numStates).
loadEnvironmentMatrices(): loads transition probability and reward matrices from files.
getProbability(): returns transition probability (not required).
getReward(): returns reward (not required).
"""

import numpy as np
import os

def __checkTransitions(env):
    """Checks if transitions add up to 1 for each state-action pair, returns False."""
    
    for i in np.sum(env.probMat, axis = 2).flat:
        if i != 1:
            return False
    
def __evaluatePolicy(env, values, policy):
    """Performs policy evaluation and returns approximate state-values for a given policy."""
    
    delta = 100
    theta = 0.01
    
    while delta > theta:
        
        delta = 0

        for s in range(0, env.numStates):
            temp = values[s]
            values[s] = np.sum(np.multiply(env.probMat[s,int(policy[s]),:], env.rewardMat[s,int(policy[s]),:] + env.gamma * values[:]))
            delta = np.maximum(delta, np.abs(temp - values[s]))

def __improvePolicy(env,values,policy):
    """Performs policy improvement on the current policy."""

    policy_stable = True

    for s in range(0, env.numStates):
        
        temp = policy[s]

        policy[s] = np.argmax(np.sum(np.multiply(env.probMat[s,:,:], env.rewardMat[s,:,:] + env.gamma * np.tile(values[:],(env.numActions,1))), axis = 1))
        
        if temp != policy[s]:
            policy_stable = False

    return policy_stable

def iteratePolicy(env):

    # make the environment load its probability and reward matrices from files
    env.loadEnvironmentMatrices()

    if not __checkTransitions(env):
        raise ValueError("Probabilities of state transitions don't add up to 1.")
    else:
        print("Transition check passed...")

    # initialize values and policy arrays
    values = np.zeros(env.numStates)
    policy = np.zeros(env.numStates)
    
    episodeCounter = 1
    while True:

        print("Episode {}".format(episodeCounter))

        # evaluate the current policy
        print("Evaluating current policy..")
        __evaluatePolicy(env,values,policy)
        
        # improve on the current policy
        print("Improving upon current policy..")
        policy_stable = __improvePolicy(env,values,policy)

        # if policy converged, return values and policy
        if policy_stable == True:
            return values, policy

        episodeCounter += 1
