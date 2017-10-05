
# Policy Iteration for Chapter 4 (Dynamic Programming)

"""
env shold have the following methods and fields:
numStates: the number of states.
numActions: the number of actions.
gamma: discount factor for the environment.
getProbability(s,a,sprime): returns the probabiliy of transition from state s to state sprime via action a.
getReward(s,a,sprime): returns the reward of going from state s to state sprime via action a.
"""

import numpy as np

def __checkTransitions(env):
    """Checks if transitions add up to 1 for each state-action pair, returns False."""
    
    for s in range(0,env.numStates):
        for a in range(0,env.numActions):
            sumProbs = 0                
            for sprime in range(0,env.numStates):
                sumProbs += env.getProbability(s,a,sprime)
                
            if sumProbs != 1:
                return False

def __evaluatePolicy(env, values, policy):
    """Performs policy evaluation and returns approximate state-values for a given policy."""
    
    delta = 100
    theta = 0.01
    
    while delta > theta:
        
        delta = 0
        
        for s in range(0, env.numStates):
            
            temp = values[s]
            newValue = 0
            for sprime in range(0, env.numStates):
                newValue += env.getProbability(s,policy[s],sprime) * (env.getReward(s,policy[s],sprime) + env.gamma * values[sprime])
                
            values[s] = newValue
            delta = np.maximum(delta, np.abs(temp - values[s]))

    return values

def __improvePolicy(env, values, policy):
    """Performs policy improvement on the current policy."""

    policy_stable = True

    for s in range(0, env.numStates):
        
        temp = policy[s]
        
        maxAction = 0
        maxActionValue = -float("inf")
        for a in range(0,env.numActions):
            newValue = 0
            for sprime in range(0, env.numStates):
                newValue += env.getProbability(s,a,sprime) * (env.getReward(s,a,sprime) + env.gamma * values[sprime])
            if newValue > maxActionValue:
                maxActionValue = newValue
                maxAction = a
                
        policy[s] = maxAction
        
        if temp != policy[s]:
            policy_stable = False

    return [policy, policy_stable]

def iteratePolicy(env):

    if __checkTransitions(env):
        raise ValueError("Probabilities of state transitions don't add up to 1.")
    else:
        print("Transition check passed...")

    # initialize variables
    values = np.zeros(env.numStates)
    policy = np.zeros(env.numStates)
    
    episodeCounter = 1
    while True:

        print("Episode {}".format(episodeCounter))

        # evaluate the current policy
        print("Evaluating current policy..")
        values =  __evaluatePolicy(env,values,policy)
        
        # improve on the current policy
        print("Improving upon current policy..")
        policy, policy_stable = __improvePolicy(env, values, policy)

        # if policy converged, return values and policy
        if policy_stable == True:
            return values, policy

        episodeCounter += 1
