
"""
Policy Iteration via state-action values for Chapter 4 (Dynamic Programming)
env shold have the following methods and fields:
numStates: the number of states.
numActions: the number of actions.
gamma: discount factor for the environment.
getQ(s,a,values): returns the value of performing action in state s
"""

import numpy as np
import matplotlib.pyplot as plt

def __evaluatePolicy(env, values, policy):
    """Performs policy evaluation and returns approximate state-values for a given policy."""
    
    delta = 100
    theta = 0.001
    
    while delta > theta:
        
        delta = 0
        
        for s in range(0, env.numStates):
            
            temp = values[s]
            values[s] = env.getQ(s,policy[s],values)                
            delta = np.maximum(delta, np.abs(temp - values[s]))

        print("Delta: {}".format(delta))

    return values

def __improvePolicy(env, values, policy):
    """Performs policy improvement on the current policy."""

    policy_stable = True

    for s in range(0, env.numStates):

        print("State {}".format(s))
        
        temp = policy[s]
        
        maxAction = 0
        maxActionValue = -float("inf")
        for a in range(0,env.numActions):
            newValue = env.getQ(s,a,values)
            if newValue > maxActionValue:
                maxActionValue = newValue
                maxAction = a
                
        policy[s] = maxAction
        
        if temp != policy[s]:
            policy_stable = False

    return [policy, policy_stable]

def iteratePolicy(env):

    # initialize variables
    values = np.zeros(env.numStates)
    policy = np.ones(env.numStates) * 5
    
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

        print("Values:\n{}".format(values))
        print("Policy:\n{}".format(np.rot90((policy-5).reshape(21,21))))
        
        plt.imshow(np.rot90((policy-5).reshape(21,21).T), cmap="hot", interpolation="nearest")
        plt.show()

        episodeCounter += 1
