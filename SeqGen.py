import numpy as np

class SequenceGenerator:
    def __init__(self, getAction, getStartState, getTransition, episodes_max=1):
        self.episodes_max = episodes_max
        self.get_action = getAction
        self.get_start_state = getStartState
        self.get_transition = getTransition

    def __iter__(self):
        self.episode_i=1
        self.state = self.get_start_state()
        return self

    def __next__(self):
        if self.episodes_max > 0 and self.episode_i > self.episodes_max:
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
        if np.random.rand(1)[0] < self.epsilon:
            return np.random.randint(0,self.Q.shape[-1])
        return np.argmax(self.Q[state])

    def getDistribution(self):
        shape = self.Q.shape
        eps_policy = np.ones(dtype=np.float, shape=shape) * self.epsilon/shape[-1]
        for state in np.ndindex(shape[:-1]):
            action = np.argmax(self.Q[state])
            eps_policy[state][action] += 1.0-self.epsilon
        return eps_policy


class SequenceGeneratorPlus:
    def __init__(self, getAction, getStartState, getTransition, episodes_max=1, steps_max=0,
                callBack=None, episode_maxlen=0):
        self.episodes_max = episodes_max
        self.get_action = getAction
        self.get_start_state = getStartState
        self.get_transition = getTransition
        self.steps_max = steps_max
        self.callback = callBack
        self.episode_maxlen = episode_maxlen
        self.episode_i = 0

    def __iter__(self):
        self.episode_i = 1
        self.state = None
        self.step_i = 0
        self.episode_step = 0
        return self

    def __next__(self):
        if self.episodes_max > 0 and self.episode_i > self.episodes_max or \
            self.steps_max > 0 and self.step_i > self.steps_max:
            raise StopIteration

        if self.state == None:
            self.state = self.get_start_state()
            self.episode_step = 0

        action = self.get_action(self.state)
        keep_state = self.state
        is_terminal, self.state, reward = self.get_transition(keep_state, action)

        if self.episode_maxlen > 0 and self.episode_step >= self.episode_maxlen:
            is_terminal = True

        if self.callback:
            self.callback(self, keep_state, is_terminal, self.state, action, reward)

        self.step_i += 1
        self.episode_i += int(is_terminal)
        self.episode_step += 1

        if is_terminal:
            self.state = None

        return keep_state, is_terminal, self.state, action, reward

