import numpy as np

def OffPolicyMCControl(episode, Q, C, getBehaviourDist, 
                       gamma=0.99, target_policy = None, CONVERGENCE_ERROR = 0.1):
    b_policy = getBehaviourDist()
    G = 0.0
    W = 1.0
    converged = True
    step = len(episode)-1
    for state, action, reward in reversed(episode):
        G = gamma*G + reward
        C[state][action] += W
        q = Q[state][action]
        Q[state][action] += W/C[state][action] * (G-Q[state][action])
        converged = converged and abs(q - Q[state][action]) < CONVERGENCE_ERROR
        new_action = np.argmax(Q[state])
        if target_policy: target_policy[state] = new_action
        if new_action != action: break
        W *= 1.0/b_policy[state][action]
        step -= 1
    
    return converged

def OnPolicyFirstVisitMCControl(episode, Q, C, gamma):
    visited = set()
    x_episode = []
    for state, action, reward in episode:
        is_first = not (state,action) in visited
        if is_first:
            visited.add((state,action))
        x_episode.append((state, action, reward, is_first))
    
    converged = False
    G = 0
    for state, action, reward, is_first in reversed(x_episode):
        G = gamma*G + reward
        if not is_first: 
            continue
        
        C[state][action] += 1
        Q[state][action] += (G-Q[state][action])/C[state][action]
    return converged

def OnPolicyEveryVisitMCControl(episode, Q, C, gamma):
    converged = False
    G = 0
    for state, action, reward in reversed(episode):
        G = gamma*G + reward
        C[state][action] += 1
        Q[state][action] += (G-Q[state][action])/C[state][action]
    
    return converged

def learnByEpisode(sequence, learnMethod):
    conv_count = 0
    episode = []

    for st,is_term,st_nx,ac,rw in sequence:
        episode.append((st,ac,rw))
        if not is_term:
            continue
        if len(episode) == 0: break
        print(" Episode #: {:7}; length: {:7}\r".format(sequence.episode_i-1, len(episode)), 
              end='', flush=True)

        if learnMethod(episode, sequence.episode_i-1):
            conv_count += 1
        else: 
            conv_count = 0
        if conv_count >= 500:
            print("\nConvergence reached.")
            break

        episode = []

    print("\nEpisodes generated: {}".format(sequence.episode_i-1))
