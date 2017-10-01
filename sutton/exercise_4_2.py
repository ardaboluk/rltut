
# Exercise 4.2 in Chapter 4 (Dynamic Programming)

import numpy as np

class PolicyIteration:
    """Class for policy iteration. Contstructor takes transition probabilities and rewards as arguments.
    Both arguments are 3-D vectors whose indexes represent state, action and new state, respectively."""

    def __init__(self, p_transitions, p_rewards):

        # shapes of both p_transitions and p_rewards are (# of states, # of actions, # of states)
        
        # check if transitions add up to 1, otherwise raise an exception
        for i in np.sum(p_transitions, axis = 2):
            if i != 1:
                raise ValueError("Probabilities doesn't add up to 1")

        self.__trans = p_transitions
        self.__rewards = p_rewards

    
