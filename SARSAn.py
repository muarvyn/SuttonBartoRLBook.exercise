import sys

REWARD_I = 4

def SARSAn(sequence, q_eval, q_learn, n, gamma):
    gamma_powered = [gamma**i for i in range(0,n+1)]
    sequence_iter = iter(sequence)
    for episode_i in range(50000):  
      try:
        step = next(sequence_iter)
        state, is_terminal, next_state, action, reward = step
        T = sys.maxsize
        t = 0
        history = [step]

        while True:
            if t < T:
                if is_terminal:
                    T = t+1
                    print("\rEpisode length is {:7}".format(T), end='', flush=True)
                else:
                    step = next(sequence_iter)
                    state, is_terminal, next_state, action, next_reward = step
                    history.append(step)

            tau = t-n+1
            if tau >= 0:
                G = sum( [gamma_powered[j]*history[tau+j][REWARD_I] for j in range(min(n,T-tau))])
                if tau+n < T:
                    G = G + gamma_powered[n] * q_eval(state, action)
                Stau, istrm, next_tau, Atau, Rtau = history[tau]
                q_learn(Stau, Atau, G)

            if tau == T-1: break
            t += 1

      except StopIteration:
        print("\nSequence terminated.")
        break

    print("\nFinished.")