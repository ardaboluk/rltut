
import numpy as np

class NSarsa:
    """n-step Sarsa in Chapter 7."""

    def __init__(self, num_states, num_actions, terminal_states, environment):

        # the environment object should provide takeAction method
        
        # number of states and actions
        self.num_states = num_states
        self.num_actions = num_actions

        # list of terminal states
        self.terminal_states = terminal_states

        # in on-policy control methods, the policy is generally soft
        # in practice, we may not store the policy because we know that
        # for E-greedy policy for example, it assigns the probability E (epsilon) 
        # to each action and 1 - E + E/|A(s)| to the greedy action
        self.policy = np.random.rand(num_states, num_actions)
        # probabilities should sum to 1
        for i in range(num_states):
            self.policy[i] = self.policy[i] / np.sum(self.policy[i])

        # values of state-action pairs
        self.q_values = np.zeros((num_states, num_actions))

        # environment object
        self.environment = environment

    def estimateQ(self, epsilon, gamma, alpha, num_steps, num_episodes):

        for episode in range(num_episodes):

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

            # store S0 and A0
            list_states.append(S0)
            list_actions.append(A0)
            # dummy reward as R0
            list_rewards.append(0)
            
            T = float('inf')
            t = 0
            tao = t - num_steps + 1
            St = S0
            At = A0
            while tao < T - 1:
                if t < T:

                    # DEBUG
                    print("state {} action {}".format(St, At))

                    # take action and observe and store the next reward and the next state
                    [next_reward, next_state] = self.environment.takeAction(St,At)
                    list_states.append(next_state)
                    list_rewards.append(next_reward)

                    St = next_state

                    next_action = None
                    if np.isin(next_state, self.terminal_states):
                        T = t + 1
                    else:
                        # select and store the next action
                        next_action = np.random.choice(self.num_actions, p = self.policy[next_state])
                        list_actions.append(next_action)
                        At = next_action                
                
                tao = t - num_steps + 1

                if tao >= 0:
                    G = 0

                    for i in range(tao + 1, np.minimum(tao + num_steps, T).astype(int) + 1):
                        G += np.power(gamma, i - tao - 1) * list_rewards[i]

                    if tao + num_steps < T:
                        G += np.power(gamma, num_steps) * self.q_values[list_states[tao + num_steps], list_actions[tao + num_steps]]

                    self.q_values[list_states[tao], list_actions[tao]] +=  alpha * (G - self.q_values[list_states[tao], list_actions[tao]])

                    # make the policy to be E-greedy w.r.t S_tao
                    max_q = np.amax(self.q_values[list_states[tao]])
                    max_q_len = np.argwhere(max_q * 1e+7 - self.q_values[list_states[tao]] * 1e+7 < 1).shape[0]
                    #max_q_len = np.argwhere(max_q == self.q_values[list_states[tao]]).shape[0]
                    greedy_prob = ((1 - epsilon) / max_q_len) + (epsilon / self.num_actions)
                    non_greedy_prob = epsilon / self.num_actions
                    for i in range(0, self.num_actions):
                        if self.q_values[list_states[tao]][i] == max_q:
                            self.policy[list_states[tao]][i] = greedy_prob
                        else:
                            self.policy[list_states[tao]][i] = non_greedy_prob

                t += 1

                # DEBUG
                #print("t: {}".format(t))
                #print("tao: {}".format(tao))                

    def getQValues(self):

        return self.q_values

    def getPolicy(self):

        return self.policy