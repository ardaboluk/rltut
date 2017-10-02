
# Exercise 4.2 in Chapter 4 (Dynamic Programming)

import numpy as np

class PolicyIteration:
    """Class for policy iteration. Contstructor takes transition probabilities and rewards as arguments.
    Both arguments of the constructor are 3-D vectors whose indexes represent state, action and new state, respectively.
    Shapes of both p_transitions and p_rewards are (numStates, numActions, numStates)."""

    def __init__(self, p_transitions, p_rewards, p_gamma):

        # check if transitions add up to 1, otherwise raise an exception
        for i in np.sum(p_transitions, axis = 2):
            if i != 1:
                raise ValueError("Exception: probabilities of transitions doesn't add up to 1")

        self.__trans = p_transitions
        self.__rewards = p_rewards
        self.__gamma = p_gamma

        self.__numStates = p_transitions.shape[0]
        self.__numActions = p_transitions.shape[1]
        
        self.__policy_probs = np.ones(self.__numStates, self.__numActions) / self.__numActions        

    def __evaluatePolicy(p_values, p_policy_probs):
        """Performs policy evaluation and calculates approximate state-values for a given policy.
        p_values: Values of states. Vector of shape (numStates).
        p_policy_probs: Probabilites of taking actions in states. Matrix of shape (numStates, numActions)."""

        delta = 0
        theta = 0.01

        while delta > theta:

            for i in range(0, p_values.shape[0]):

                temp = p_values[i]
                p_values[i] = np.multiply(np.tile(self.__policy_probs[i,:], (1,1,self.__numStates)), self.__trans[i,:,:], (self.__rewards[i,:,:] + self.__gamma * np.tile(1,self.__numActions,1)))
                delta = np.maximum(delta, np.abs(temp - p_values[i]))
                
                
