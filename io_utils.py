import numpy as np
from scipy import stats
import os
import glob
import sklearn.preprocessing
import argparse

jerk_threshold = 100.0 #0.0375

def get_filelist(dirname, savebinary=False):
    if savebinary:
        filelist = glob.glob(dirname + '/*.npy')
    else:
        filelist = glob.glob(dirname + '/*.csv')
    filelist.sort()
    return np.array(filelist)

def get_data(dirname, skip=0, include_time=False):
    filelist = get_filelist(dirname, savebinary=False)

    data = []
    for filename in filelist:
        try:
            tmp = np.loadtxt(filename, delimiter=',')
        except Exception as e:
            print("file load error@",filename)
            print(e)
            return None
        if include_time:
            data.append(tmp[skip:])
        else:
            data.append(tmp[skip:][:, 1:])

    return np.array(data), filelist

def get_targets(rootdirname, patientname, classname, skip=0, include_time=False):
    
    filename = os.path.join(rootdirname+'/'+patientname+'/'+classname+'.csv')

    data = []
    try:
        tmp = np.loadtxt(filename, delimiter=',')
    except Exception as e:
        print("file load error@",filename)
        print(e)
        return None
    if include_time:
        data.append(tmp[skip:])
    else:
        data.append(tmp[skip:][:, 1:])

    return np.array(data).reshape((-1))

def get_data_v2(dirname, patientname, skip=0, include_time=False):
    
    filelist = get_filelist(dirname, savebinary=False)

    data = []
    file_list = []
    for filename in filelist:
        basefilename = os.path.basename(filename)
        if basefilename[:len(patientname)] == patientname:
            try:
                tmp = np.loadtxt(filename, delimiter=',')
                file_list.append(filename)
            except Exception as e:
                print("file load error@",filename)
                print(e)
                return None
            if include_time:
                data.append(tmp[skip:])
            else:
                data.append(tmp[skip:][:, 1:])

    return np.array(data), file_list

def get_data_v3(dirname, patientname, include_jerk, skip=0, include_time=False):
    
    filelist = get_filelist(dirname, savebinary=False)
    data = []
    file_list = []
    for filename in filelist:
        basefilename = os.path.basename(filename)
        if basefilename[:len(patientname)] == patientname:
            try:
                tmp = np.loadtxt(filename, delimiter=',')
                file_list.append(filename)
            except Exception as e:
                print("file load error@",filename)
                print(e)
                return None
            if include_jerk:
                if tmp.shape[1] > 3:
                    max_jerk = np.max(tmp[:,3])
                elif tmp.shape[1] > 2:
                    max_jerk = np.max(tmp[:,2])
                else:
                    max_jerk = np.max(tmp[:,1])

                if max_jerk <= jerk_threshold:
                    if include_time:
                        data.append(tmp[skip:])
                    else:
                        data.append(tmp[skip:][:, 1:])

    return np.array(data), file_list

def get_data_v4(dirname, patientname, feature_type, skip=0, include_time=False):
    
    filelist = get_filelist(dirname, savebinary=False)
    data = []
    file_list = []
    for filename in filelist:
        basefilename = os.path.basename(filename)
        if basefilename[:len(patientname)] == patientname and basefilename[len(patientname):len(patientname)+1] == '_':
            try:
                tmp = np.loadtxt(filename, delimiter=',')
                if len(tmp) < 1:
                    continue
            except Exception as e:
                print("file load error@",filename)
                print(e)
                return None
            max_jerk = np.max(tmp[:,3])

            if max_jerk <= jerk_threshold:
                file_list.append(filename)
                if feature_type == 'speed':
                    if include_time:
                        data.append(tmp[skip:][:, [0,1]])
                    else:
                        data.append(tmp[skip:][:, 1:2])
                elif feature_type == 'acceleration':
                    if include_time:
                        data.append(tmp[skip:][:, [0,2]])
                    else:
                        data.append(tmp[skip:][:, 2:3])
                elif feature_type == 'jerk':
                    if include_time:
                        data.append(tmp[skip:][:, [0,3]])
                    else:
                        data.append(tmp[skip:][:, 3:])
                elif feature_type == 'speed and acceleration':
                    if include_time:
                        data.append(tmp[skip:][:, [0,1,2]])
                    else:
                        data.append(tmp[skip:][:, [1,2]])
                elif feature_type == 'speed and jerk':
                    if include_time:
                        data.append(tmp[skip:][:, [0,1,3]])
                    else:
                        data.append(tmp[skip:][:, [1,3]])
                elif feature_type == 'acceleration and jerk':
                    if include_time:
                        data.append(tmp[skip:][:, [0,2,3]])
                    else:
                        data.append(tmp[skip:][:, [2,3]])
                else:
                    if include_time:
                        data.append(tmp[skip:])
                    else:
                        data.append(tmp[skip:][:, 1:])
                        
    return np.array(data), file_list

def get_data_v4_2(dirname, feature_type, skip=0, include_time=False):

    filelist = get_filelist(dirname, savebinary=False)

    data = []
    for filename in filelist:
        try:
            tmp = np.loadtxt(filename, delimiter=',')
        except Exception as e:
            print("file load error@",filename)
            print(e)
            return None
        if feature_type == 'speed':
            if include_time:
                data.append(tmp[skip:][:, [0,1]])
            else:
                data.append(tmp[skip:][:, 1:2])
        elif feature_type == 'acceleration':
            if include_time:
                data.append(tmp[skip:][:, [0,2]])
            else:
                data.append(tmp[skip:][:, 2:3])
        elif feature_type == 'jerk':
            if include_time:
                data.append(tmp[skip:][:, [0,3]])
            else:
                data.append(tmp[skip:][:, 3:])
        elif feature_type == 'speed and acceleration':
            if include_time:
                data.append(tmp[skip:][:, [0,1,2]])
            else:
                data.append(tmp[skip:][:, [1,2]])
        elif feature_type == 'speed and jerk':
            if include_time:
                data.append(tmp[skip:][:, [0,1,3]])
            else:
                data.append(tmp[skip:][:, [1,3]])
        elif feature_type == 'acceleration and jerk':
            if include_time:
                data.append(tmp[skip:][:, [0,2,3]])
            else:
                data.append(tmp[skip:][:, [2,3]])
        else:
            if include_time:
                data.append(tmp[skip:])
            else:
                data.append(tmp[skip:][:, 1:])

    return np.array(data), filelist

def get_one_data(filename, skip=0, include_time=False, ignore_header=True):
    if not filename.endswith(".npy"):
        tmp = np.loadtxt(filename, delimiter=',') if not ignore_header else np.loadtxt(filename, delimiter=',',skiprows=1)
    else:
        tmp = np.load(filename)
    if include_time:
        return np.array(tmp[skip:])
    else:
        return np.array(tmp[skip:][:, 1:])
        
            
def get_max_length(normal_data, mutant_data):
    length = np.array([])
    for data in normal_data:
        length = np.append(length, len(data))
    for data in mutant_data:
        length = np.append(length, len(data))

    return int(np.max(length))

def standardization(input_array, mean, std, bias=0.0):
    return ((input_array - mean) / np.maximum(std, 10 ** -5)) + bias

def normalization_zero_one(input_array, cur_min, cur_max):
    return (input_array - cur_min) / (cur_max - cur_min)

def normalization_min_max(input_array, cur_min, cur_max, new_min, new_max):
    return new_min + (((input_array - cur_min) * (new_max - new_min)) / (cur_max - cur_min))

def normalize_list(normal_list, mutant_list, bias=0.0):
    nc = np.concatenate(normal_list)
    mc = np.concatenate(mutant_list)
    con = np.concatenate((nc, mc))

    mean = np.mean(con, axis=0)
    std = np.std(con, axis=0)

    ret_normal = []
    ret_mutant = []
    for i in range(len(normal_list)):
        ret_normal.append(standardization(normal_list[i], mean, std, bias=bias))
    for i in range(len(mutant_list)):
        ret_mutant.append(standardization(mutant_list[i], mean, std, bias=bias))

    return np.array(ret_normal), np.array(ret_mutant), mean, std

def normalize_list_v21(normal_list, mutant_list, bias=0.0):
    nc = np.concatenate(normal_list)
    mc = np.concatenate(mutant_list)
    con = np.concatenate((nc, mc))

    mean = np.mean(con, axis=0)
    std = np.std(con, axis=0)

    ret_normal = []
    ret_mutant = []
    for i in range(len(normal_list)):
        ret_normal.append(standardization(normal_list[i], mean, std, bias=bias))
    for i in range(len(mutant_list)):
        ret_mutant.append(standardization(mutant_list[i], mean, std, bias=bias))

    return np.array(ret_normal), np.array(ret_mutant), mean, std

def normalize_list_v2(normal_list, mutant_list, mutant1_list, mutant2_list, mutant3_list, bias=0.0):
    nc = np.concatenate(normal_list)
    mc = np.concatenate(mutant_list)
    mc1 = np.concatenate(mutant1_list)
    mc2 = np.concatenate(mutant2_list)
    mc3 = np.concatenate(mutant3_list)
    con = np.concatenate((nc, mc, mc1, mc2, mc3))

    mean = np.mean(con, axis=0)
    std = np.std(con, axis=0)

    ret_normal = []
    ret_mutant = []
    ret_mutant1 = []
    ret_mutant2 = []
    ret_mutant3 = []
    for i in range(len(normal_list)):
        ret_normal.append(standardization(normal_list[i], mean, std, bias=bias))
    for i in range(len(mutant_list)):
        ret_mutant.append(standardization(mutant_list[i], mean, std, bias=bias))
    for i in range(len(mutant1_list)):
        ret_mutant1.append(standardization(mutant1_list[i], mean, std, bias=bias))
    for i in range(len(mutant2_list)):
        ret_mutant2.append(standardization(mutant2_list[i], mean, std, bias=bias))
    for i in range(len(mutant3_list)):
        ret_mutant3.append(standardization(mutant3_list[i], mean, std, bias=bias))

    return np.array(ret_normal), np.array(ret_mutant), np.array(ret_mutant1), np.array(ret_mutant2), np.array(ret_mutant3), mean, std

def normalize_list_v3(normal_list, mutant_list, mutant1_list, mutant2_list, mutant3_list, bias=0.0):

    con = np.concatenate((normal_list, mutant_list, mutant1_list, mutant2_list, mutant3_list))

    mean = np.mean(con, axis=0)
    std = np.std(con, axis=0)

    ret_normal = []
    ret_mutant = []
    ret_mutant1 = []
    ret_mutant2 = []
    ret_mutant3 = []
    for i in range(len(normal_list)):
        ret_normal.append(standardization(normal_list[i], mean, std, bias=bias))
    for i in range(len(mutant_list)):
        ret_mutant.append(standardization(mutant_list[i], mean, std, bias=bias))
    for i in range(len(mutant1_list)):
        ret_mutant1.append(standardization(mutant1_list[i], mean, std, bias=bias))
    for i in range(len(mutant2_list)):
        ret_mutant2.append(standardization(mutant2_list[i], mean, std, bias=bias))
    for i in range(len(mutant3_list)):
        ret_mutant3.append(standardization(mutant3_list[i], mean, std, bias=bias))

    return np.array(ret_normal), np.array(ret_mutant), np.array(ret_mutant1), np.array(ret_mutant2), np.array(ret_mutant3), mean, std

def normalize_list_v31(normal_list, mutant_list, bias=0.0):

    con = np.concatenate((normal_list, mutant_list))

    mean = np.mean(con, axis=0)
    std = np.std(con, axis=0)

    ret_normal = []
    ret_mutant = []
    for i in range(len(normal_list)):
        ret_normal.append(standardization(normal_list[i], mean, std, bias=bias))
    for i in range(len(mutant_list)):
        ret_mutant.append(standardization(mutant_list[i], mean, std, bias=bias))

    return np.array(ret_normal), np.array(ret_mutant), mean, std

def normalize_list_zero_one(normal_list, mutant_list, mutant1_list, mutant2_list, mutant3_list):

    con = np.concatenate((normal_list, mutant_list, mutant1_list, mutant2_list, mutant3_list))

    cur_min = np.min(con, axis=0)
    cur_max = np.max(con, axis=0)

    ret_normal = []
    ret_mutant = []
    ret_mutant1 = []
    ret_mutant2 = []
    ret_mutant3 = []
    for i in range(len(normal_list)):
        ret_normal.append(standardization(normal_list[i], cur_min, cur_max))
    for i in range(len(mutant_list)):
        ret_mutant.append(standardization(mutant_list[i], cur_min, cur_max))
    for i in range(len(mutant1_list)):
        ret_mutant1.append(standardization(mutant1_list[i], cur_min, cur_max))
    for i in range(len(mutant2_list)):
        ret_mutant2.append(standardization(mutant2_list[i], cur_min, cur_max))
    for i in range(len(mutant3_list)):
        ret_mutant3.append(standardization(mutant3_list[i], cur_min, cur_max))

    return np.array(ret_normal), np.array(ret_mutant), np.array(ret_mutant1), np.array(ret_mutant2), np.array(ret_mutant3)

def normalize_list_min_max(normal_list, mutant_list, mutant1_list, mutant2_list, mutant3_list, new_min, new_max):

    con = np.concatenate((normal_list, mutant_list, mutant1_list, mutant2_list, mutant3_list))

    cur_min = np.min(con, axis=0)
    cur_max = np.max(con, axis=0)

    ret_normal = []
    ret_mutant = []
    ret_mutant1 = []
    ret_mutant2 = []
    ret_mutant3 = []
    for i in range(len(normal_list)):
        ret_normal.append(standardization(normal_list[i], cur_min, cur_max, new_min, new_max))
    for i in range(len(mutant_list)):
        ret_mutant.append(standardization(mutant_list[i], cur_min, cur_max, new_min, new_max))
    for i in range(len(mutant1_list)):
        ret_mutant1.append(standardization(mutant1_list[i], cur_min, cur_max, new_min, new_max))
    for i in range(len(mutant2_list)):
        ret_mutant2.append(standardization(mutant2_list[i], cur_min, cur_max, new_min, new_max))
    for i in range(len(mutant3_list)):
        ret_mutant3.append(standardization(mutant3_list[i], cur_min, cur_max, new_min, new_max))

    return np.array(ret_normal), np.array(ret_mutant), np.array(ret_mutant1), np.array(ret_mutant2), np.array(ret_mutant3)

def normalize_one(target_list, mean, std, bias=0.0):
    ret_target = []
    for i in range(len(target_list)):
        ret_target.append(standardization(target_list[i], mean, std, bias=bias))
    return ret_target

def normalize_one_v2(target_list, bias=0.0):
    tc = np.concatenate(target_list)

    mean = np.mean(tc, axis=0)
    std = np.std(tc, axis=0)

    ret_target = []
    for i in range(len(target_list)):
        ret_target.append(standardization(target_list[i], mean, std, bias=bias))
    return np.array(ret_target), mean, std


def splitData_by_random(data, percentage, seed=0):
    np.random.seed(seed)
    index = np.random.permutation(len(data))

    train = data[index[:int(len(data) * percentage)]]
    test = data[index[int(len(data) * percentage):]]

    return train, test

def splitData_by_random_v2(data, targets, percentage, seed=0):
    np.random.seed(seed)
    index = np.random.permutation(len(data))

    train = data[index[:int(len(data) * percentage)]]
    test = data[index[int(len(data) * percentage):]]

    target_train = targets[index[:int(len(data) * percentage)]]
    target_test = targets[index[int(len(data) * percentage):]]

    return train, test, target_train, target_test


def splitData_by_index(data, splited_indexes):
    splited_data = np.array([data[index] for index in splited_indexes])
    return splited_data



def hotvec(dim, label, num):
    ret = []
    for i in range(num):
        vec = [0] * dim
        vec[label] = 1
        ret.append(vec)
    return np.array(ret)


def hotvec_time(dim, label, num, time):
    ret = []
    for i in range(num):
        vec = [0] * dim
        vec[label] = 1
        ret.append([vec for j in range(time)])
    return np.array(ret)
