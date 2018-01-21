
import numpy as np

class NSarsa:
    """n-step Sarsa in Chapter 7."""

    def __init__(self, num_states, num_actions, terminal_states, take_action_callback):
        
        # number of states and actions
        self.num_states = num_states
        self.num_actions = num_actions

        # list of terminal states
        self.terminal_states = terminal_states

        # in on-policy control methods, the policy is generally soft
        # in practice, we may not store the policy because we know that
        # for E-greedy policy for example, it assigns the probability E (epsilon) 
        # to each action and 1 - E + E/|A(s)| to the greedy action
        self.policy = np.ndarray(shape=(num_states, num_actions))

        # values of state-action pairs
        self.q_values = np.ndarray(shape=(num_states, num_actions))

        # callback for the function which tells the reward and the next state for a given state-action pair
        self.tac = take_action_callback

    def estimateQ(self, epsilon, gamma, alpha, num_steps, num_episodes):

        for episode in num_episodes:

            # list of states, actions and rewards observed through the episode
            list_states = []
            list_actions = []
            list_rewards = []            

            # select a random state other than a terminal state
            S0 = np.random.randint(self.num_states)
            while np.isin(S0,self.terminal_states):
                S0 = np.random.randint(self.num_states)

            # select an action from the E-greedy policy
            A0 = np.random.choice(self.num_actions, p = self.policy[S0])
            
            T = float('inf')
            t = 0
            tao = t - num_steps + 1
            St = S0
            At = A0
            while tao < T - 1:
                if t < T:
                    [next_reward, next_state] = self.tac(St,At)
                    list_states.append(St)
                    list_actions.append(At)
                    list_rewards.append(next_reward)

                    next_action = None
                    if np.isin(next_state, self.terminal_states):
                        T = t + 1
                    else:
                        next_action = np.random.choice(self.num_actions, p = self.policy(next_state))
                
                tao = t - num_steps + 1

                if tao >= 0:
                    G = 0

                    for i in range(tao + 1, np.minimum(tao + n, T) + 1):
                        G += np.power(gamma, i - tao - 1) * list_rewards[i]

                    if tao + n < T:
                        G += np.power(gamma, num_steps) * self.q_values[list_states[tao + num_steps], list_actions[tao + num_steps]]

                    self.q_values[list_states[tao], list_actions[tao]] +=  alpha * (G - self.q_values[list_states[tao], list_actions[tao]])

                    

                t += 1        