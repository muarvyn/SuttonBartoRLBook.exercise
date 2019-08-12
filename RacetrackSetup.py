import numpy as np

MAX_VELOCITY = 4
MIN_VELOCITY = 0

# Setup #1
contour_1 = [
    [3,0],[3,3],[2,3],[2,10],[1,10],[1,18],[0,18],[0,28],[1,28],[1,29],[2,29],
    [2,31],[3,31],[3,32],[17,32],[17,26],[10,26],[10,25],[9,25],[9,0],[3,0]
]
start_line_1 = (3,0,9,1)
finish_line_1 = (16,26,17,32)
track_shape_1 = (17,32)

# Setup #2
contour_2 = [
    [0,0],[0,3],[1,3],[1,4],[2,4],[2,5],[3,5],[3,6],[4,6],[4,7],[5,7],[5,8],
    [6,8],[6,9],[7,9],[7,10],[8,10],[8,11],[9,11],[9,12],[10,12],[10,13],[11,13],[11,14],
    [12,14],[12,15],[13,15],[13,16],[14,16],[14,21],[13,21],[13,22],[12,22],[12,23],[11,23],[11,27],
    
    [12,27],[12,28],[13,28],[13,29],[16,29],[16,30],[32,30],[32,21],[30,21],[30,20],[27,20],[27,19],
    [26,19],[26,18],[24,18],[24,17],[23,17],[23,0],[0,0]
]
start_line_2 = (0,0,23,1)
finish_line_2 = (31,21,32,30)
track_shape_2 = (32,30)

ACCELERATION =  [[1,-1],  [1,0],  [1,1],
                 [0,-1],  [0,0],  [0,1],
                [-1,-1], [-1,0], [-1,1]]
ACTIONS_NUM = len(ACCELERATION)

def getRelative(base, line):
    return line-base
    
def det(v1,v2):
    return v1[0]*v2[1]-v1[1]*v2[0]

def is_intersection(l1, l2):
    return (det(*getRelative(l1[0],l2)) < 0) != (det(*getRelative(l1[1],l2)) < 0) \
           and (det(*getRelative(l2[0],l1)) < 0) != (det(*getRelative(l2[1],l1)) < 0)

def getStartPosition(start_line):
    return (np.random.randint(start_line[0], start_line[2]),0,0,0)

def is_finished(finish_line, position, next_position):
    return is_intersection(np.array([(finish_line[0],finish_line[1]),
                                     (finish_line[0],finish_line[3])]),
                           np.array([np.array(position)+0.5, 
                                     np.array(next_position)+0.5]))

def is_runout(contour, position, next_position):
    it = iter(contour)
    p1 = next(it)
    for p2 in it:
        if is_intersection(np.array([p1,p2]), 
                           np.array([np.array(position)+0.5, 
                                     np.array(next_position)+0.5])):
            return True
        p1 = p2
    return False

def getTransition(track_contour, state, action, finish_line, getStartPosition):
    accel = ACCELERATION[action]
    # challanging non-deterministic behavior
    #if np.random.randint(0,10,1)[0] == 0:
    #    accel = [0,0]

    next_velocity = np.clip([state[2]+accel[0],state[3]+accel[1]], 
                            MIN_VELOCITY, MAX_VELOCITY)
    if (next_velocity == 0).all():
        next_velocity = np.array([state[2],state[3]])
    position = np.array([state[0],state[1]])
    next_position = position + next_velocity
    next_state = tuple(next_position) + tuple(next_velocity)

    if is_finished(finish_line, position, next_position):
        return (True,next_state)

    if is_runout(track_contour, position, next_position):
        next_state = getStartPosition()

    return (False,next_state)
