
# Exercise 4.2 in Chapter 4 (Dynamic Programming)

import numpy as np

# set the environment

transitions = np.ones(21,5,21)
rewards = np.zeros(21,5,21)
gamma = 0.9



class PolicyIteration:
    """Class for policy iteration. Contstructor takes transition probabilities and rewards as arguments.
    Both arguments of the constructor are 3-D vectors whose indexes represent state, action and new state, respectively.
    Shapes of both p_transitions and p_rewards are (numStates, numActions, numStates)."""

    def __init__(self, p_transitions, p_rewards, p_gamma):

        # check if transitions add up to 1, otherwise raise an exception
        for i in np.sum(p_transitions, axis = 2):
            if i != 1:
                raise ValueError("Exception: probabilities of transitions doesn't add up to 1")

        self.__transitions = p_transitions
        self.__rewards = p_rewards
        self.__gamma = p_gamma

        self.__numStates = p_transitions.shape[0]
        self.__numActions = p_transitions.shape[1]

        # Policy is indexes of actions for each state. Vector of shape (numStates)
        self.__policy = np.zeros(self.__numStates)
        # Values of states. Vector of shape (numStates)
        self.__s_values = np.zeros(self.__numStates)
        # useful in checking if the policy has been improved further
        self.__policy_stable = False

    def __evaluatePolicy():
        """Performs policy evaluation and calculates approximate state-values for a given policy.
        p_values: """

        delta = 100
        theta = 0.01

        while delta > theta:

            delta = 0

            for i in range(0, self.__numStates):

                temp = self.__s_values[i]
                p_values[i] = np.sum(np.multiply(self.__transitions[i, self.__policy[i], :], (self.__rewards[i, self.__policy[i], :] + self.__gamma * self.__s_values)))
                delta = np.maximum(delta, np.abs(temp - p_values[i]))

    def __improvePolicy():
        """Performs policy improvement on the current policy."""

        self.__policy_stable = True

        for i in range(0, self.__numStates):

            temp = self.__policy[i]
            self.__policy[i] = np.argmax(np.sum(np.multiply(self.__transitions[i,:,:], (self.__rewards[i,:,:] + self.__gamma * np.tile(self.__s_values, (self.__numActions,1)))), axis = 2))
            if temp != self.__policy[i]:
                self.__policy_stable = False
            
                
    def optimize():
        """Finds the approximate optimal policy for a given MDP."""

        while self.__policy_stable == False:

            self.__evaluatePolicy()
            self.__improvePolicy()

        return [self.__s_values, self.__policy]
