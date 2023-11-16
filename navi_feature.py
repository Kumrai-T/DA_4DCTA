import numpy as np
from ipywidgets import FloatProgress
from keras.preprocessing import sequence
from sklearn import preprocessing
import random
import pandas as pd

# sign of angle
def sign_angle(p1, p2):
    sign = np.cross(p1, p2)
    if sign > 0:
        sign = 1
    else:
        sign = -1

    return sign


# calculate relative angle
def angle(p1, p2):
    x1 = np.array(p1, dtype=np.float)
    x2 = np.array(p2, dtype=np.float)

    Lx1 = np.sqrt(x1.dot(x1))
    Lx2 = np.sqrt(x2.dot(x2))

    if Lx1 * Lx2 == 0:
        angle_abs = 0
        return 0
    elif round(x1[0] * x2[1] - x1[1] * x2[0], 4) == 0.0:
        if x1[0] != 0:
            if (x1[0] * x2[0] > 0.0):
                angle_abs = 0
            else:
                angle_abs = -np.pi
        else:
            if (x1[1] * x2[1] > 0.0):
                angle_abs = 0
            else:
                angle_abs = -np.pi
    else:
        cos_ang = x1.dot(x2) / (Lx1 * Lx2)
        angle_abs = np.arccos(cos_ang)

    ##    print x1, x2, x1[0]*x2[1], x1[1]*x2[0], angle_abs

    sign = sign_angle(x1, x2)

    return sign * angle_abs


# each arc-length
def length(p):
    x = np.array(p)
    Lp = np.sqrt(x.dot(x))
    return Lp


# iterative angle normalizaton(fix relative angle oscillate aroud Pi)
def angle_normalization(angle):
    avg_angle = np.average(angle)
    for i in range(10):
        if avg_angle != np.average(angle):
            break
        avg_angle = np.average(angle)

        for j in range(len(angle)):
            angle[j] = angle[j] - avg_angle

        for j in range(len(angle)):
            if angle[j] > np.pi:
                ##                print j, angle[j]
                angle[j] = -2 * np.pi + angle[j]
            if angle[j] < -np.pi:
                ##                print j, angle[j]
                angle[j] = 2 * np.pi + angle[j]

                ##        print avg_angle
    return 0


def acceleration(p0, p1, p2, timediff0, timediff1):
    vec0 = np.array([p1[0] - p0[0], p1[1] - p0[1]]) / timediff0
    vec1 = np.array([p2[0] - p1[0], p2[1] - p1[1]]) / timediff1

    accvec = [vec1[0] - vec0[0], vec1[1] - vec0[1]]

    return length(accvec), np.arctan2(accvec[1], accvec[0])

def primitive_feature_extraction(time, x, y):
    timelist = np.array([])
    speedlist = np.array([])
    accelerationlist = np.array([])
    accelerationlist_angle = np.array([])
    relative_anglelist = np.array([])
    anglelist = np.array([])
    line_distance_from_init = np.array([])
    angle_from_init = np.array([])
    travel_distance_from_init = np.array([])
    i0 = -1
    for i in range(len(time)):
        if i < 2:
            continue
        else:
            if i0 == -1:
                i0 = i
            timelist = np.append(timelist, time[i])
            timediff = float(time[i] - time[i - 1])
            v_t = []
            # angle relative to previous
            v_ref = [x[i - 1] - x[i - 2], y[i - 1] - y[i - 2]]

            v_t.append(x[i] - x[i - 1])
            v_t.append(y[i] - y[i - 1])

            angle_t = angle(v_t, v_ref)
            relative_anglelist = np.append(relative_anglelist, angle_t / timediff)
            speedlist = np.append(speedlist, length(v_t) / timediff)

            acc, accangle = acceleration([x[i - 2], y[i - 2]], [x[i - 1], y[i - 1]], [x[i], y[i]],
                                         time[i - 1] - time[i - 2], timediff)
            accelerationlist = np.append(accelerationlist, acc)
            accelerationlist_angle = np.append(accelerationlist_angle, accangle)

            anglelist = np.append(anglelist, np.arctan2(v_t[1], v_t[0]) / timediff)

            # distance from initial point
            line_distance_from_init = np.append(line_distance_from_init, length(
                [x[i] - x[i0], y[i] - y[i0]]))
            # angle from initial point
            angle_from_init = np.append(angle_from_init, np.arctan2(
                y[i] - y[i0], x[i] - x[i0]))
            # travel distance
            if len(travel_distance_from_init) == 0:
                travel_distance_from_init = np.append(travel_distance_from_init, 0)
            else:
                travel_distance_from_init = np.append(travel_distance_from_init,
                                                      length(v_t) + travel_distance_from_init[-1])

    angle_normalization(relative_anglelist)
    abs_relative_anglelist = np.abs(relative_anglelist)
    df = pd.DataFrame({'time':timelist, 'speed':speedlist, 'acceleration':accelerationlist, 'acc_angle':accelerationlist_angle
                      , 'rel_angle':relative_anglelist, 'abs_rel_angle':abs_relative_anglelist, 'angle':anglelist, 'straight_line_dist_from_init':line_distance_from_init
                      , 'travel_distance_from_init':travel_distance_from_init, 'angle_from_init':angle_from_init})
    return df

def primitive_feature_extraction_all(dfs, time_col='time', x_col='x', y_col='y'):
    result = []
    fp = FloatProgress(min=0, max=len(dfs))
    fp.value = 0
    display(fp)
    for df in dfs:
        df_new = pd.merge(df, primitive_feature_extraction(df[time_col], df[x_col], df[y_col]))
        result.append(df_new)
        fp.value = fp.value + 1
    return result

def standardization(input_array, mean, std, bias=0.0):
    return ((input_array - mean) / np.maximum(std, 10 ** -5)) + bias

def to_nparray(df, features=['speed','abs_rel_angle']):
    arr = df[features].values
    return arr

def to_nparrays(dfs, features=['speed','abs_rel_angle'], shuffle = False, standardize = True):
    X = []
    max_len = 0
    for df in dfs:
        arr = to_nparray(df, features)
        max_len = max([max_len, len(arr)])
        X.append(arr)
    if shuffle: 
        random.shuffle(X)
    if standardize:
        con = np.concatenate(X)
        mean = np.mean(con, axis=0)
        std = np.std(con, axis=0)
        print("mean:",mean)
        print("std:",std)
        _X = []
        for x in X:
            _X.append(standardization(x, mean, std)) 
        X = _X
        
    X = sequence.pad_sequences(X, maxlen=max_len, dtype='float64', padding='post',
                                            truncating='post', value=-1.0)
    return X
