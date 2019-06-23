import numpy as np

def getStartPosition():
    return (np.random.rand(1)[0]*0.2-0.6, 0.0)

REWARD = -1.0

def getTransition(position, velocity, push):
    finished = False
    v = np.clip(velocity + 0.001*push - 0.0025*np.cos(3.0*position), -0.07, 0.07)
    p = position + v
    if p <= -1.2:
        p = -1.2
        v = 0.0
    elif p >= 0.5:
        p = 0.5
        finished = True
    return (p, v, finished, REWARD)