import numpy as np

class SequenceGenerator:
    def __init__(self, getAction, getStartState, getTransition, episode_imax=1):
        self.episode_imax = episode_imax
        self.get_action = getAction
        self.get_start_state = getStartState
        self.get_transition = getTransition

    def __iter__(self):
        self.episode_i=1
        self.state = self.get_start_state()
        return self

    def __next__(self):
        if self.episode_imax > 0 and self.episode_i > self.episode_imax:
            raise StopIteration

        action = self.get_action(self.state, self.episode_i)
        keep_state = self.state
        is_terminal, self.state, reward = self.get_transition(keep_state, action)
        self.episode_i += int(is_terminal)
        return keep_state, is_terminal, self.state, action, reward

REWARD_I = 4
STATE_I = 0

class EpsilonGreedyPolicy:
    def __init__(self, Q, Epsilon=0.1):
        self.Q = Q;
        self.epsilon = Epsilon

    def __call__(self, state, episode_i=1):
        q = self.Q[state]
        if np.random.rand(1)[0] < self.epsilon:
            return np.random.randint(0,len(q))
        return np.argmax(q)

