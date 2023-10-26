import os
import pandas as pd
import math
import numpy as np
import navi_feature


def feature_extraction(csvfile, feat_class_dir, patient_name, args):
    df = pd.read_csv(csvfile)
    if "extractor" in args:
        if args["extractor"] =="ujisawa":
            feature_extraction_ujisawa(csvfile, feat_class_dir, args)
            return
        
    ncol = len(df.columns)
    if ncol == 4:
        feature_extraction_with_time_3d_speed(csvfile, feat_class_dir, patient_name, args)
    else:
        feature_extraction_without_time_3d_speed(csvfile, feat_class_dir, args, time_interval=0.0115)

def feature_extraction_ujisawa(csvfile, feat_class_dir, args):     
    header_names = ["time", "temperature", "x", "y"]
    df = pd.read_csv(csvfile, names=header_names, header=1)
    df = df.dropna(how='any', axis=0)
    #display(df.head())
    time = df["time"].values
    x = df["x"].values
    y = df["y"].values
    feat_df = navi_feature.primitive_feature_extraction(time, x, y)
    feat_df = feat_df.drop("angle_from_init", axis=1)
    df = df.drop("x", axis=1)
    df = df.drop("y", axis=1)
    
    feat_df = pd.merge(feat_df, df, on="time")
    feat_df = feat_df.dropna(how='any', axis=1)
    feat_df.reset_index(drop=True, inplace=True)
    feat_df = feat_df.rename(columns={'time': '# time'})
    #display(feat_df.head())
    dfsize = 1000
    if "size" in args:
        dfsize = args["size"]
    dfs = slice_df(feat_df, dfsize)
    dfs.pop(-1)
    
    for i,one_df in enumerate(dfs):
        savefilename = os.path.basename(csvfile).replace(".csv", "_"+str(i).zfill(3)+".csv")
        one_df.to_csv(os.path.join(feat_class_dir, savefilename), index=False)
        
        
def slice_df(df: pd.DataFrame, size: int) -> list:
    """pandas.DataFrameを行数sizeずつにスライスしてリストに入れて返す"""
    previous_index = list(df.index)
    df = df.reset_index(drop=True)
    n = df.shape[0]
    list_indices = [(i, i+size) for i in range(0, n, size)]
    df_indices = [(i, i+size-1) for i in range(0, n, size)]
    sliced_dfs = []
    for i in range(len(df_indices)):
        begin_i, end_i = df_indices[i][0], df_indices[i][1]
        begin_l, end_l = list_indices[i][0], list_indices[i][1]
        df_i = df.loc[begin_i:end_i, :]
        df_i.index = previous_index[begin_l:end_l]
        sliced_dfs += [df_i]
    return sliced_dfs

def feature_extraction_with_time_3d_speed(csvfile, feat_class_dir, patient_name, args):
    use_header = True
    header_names = ["time", "x", "y", "z"]
    if "use_header" in args:
        use_header = args["use_header"]
        if "header_names" in args:
            header_names = args["header_names"]
    if use_header:
        df = pd.read_csv(csvfile)
    else:
        df = pd.read_csv(csvfile, header=None, names=header_names)
    p_time = p_x = p_y = p_z = 0
    p_time2 = p_x2 = p_y2 = p_z2 = 0
    timelist = []
    speedlist = []
    xlist, ylist, zlist = [], [], []
    for index, row in df.iterrows():
        time = row[0]
        x = row[1]
        y = row[2]
        z = row[3]
        xlist.append(x)
        ylist.append(y)
        zlist.append(z)
        if index > 0:
            timelist.append(time)
            dist = euc_dist_3d(x, y, z, p_x, p_y, p_z)
            speed = dist / (time - p_time)
            speedlist.append(speed)
        else:
            timelist.append(0)
            speedlist.append(0)
        p_time2 = p_time
        p_x2 = p_x
        p_y2 = p_y
        p_z2 = p_z
        p_time = time
        p_x = x
        p_y = y
        p_z = z

    accelerationlist = []
    for i in range(len(speedlist)):
        if i > 0:
            t1, t2 = timelist[i-1], timelist[i]
            v1, v2 = speedlist[i-1], speedlist[i]
            dv = v2 - v1
            dt = t2 - t1
            a = dv / dt
            accelerationlist.append(a)
        else:
            accelerationlist.append(0)
    jerklist = []
    for i in range(len(accelerationlist)):
        if i > 0:
            t1, t2 = timelist[i-1], timelist[i]
            a1, a2 = accelerationlist[i-1], accelerationlist[i]    
            da = a2 - a1
            dt = t2 - t1
            j = da / dt    
            jerklist.append(j)
        else:
            jerklist.append(0)

    if "training" in args:
        training_list = args["training"]
    if "testing" in args:
        testing_list = args["testing"]

    if patient_name in training_list:
        last_folder_name = os.path.basename(feat_class_dir)
        bw_dir = os.path.normpath(feat_class_dir + os.sep + os.pardir)
        bw_folder_name = os.path.basename(bw_dir)
        savefilename = patient_name + '_' + os.path.basename(csvfile)
        save_feat_class_dir = feat_class_dir

    if patient_name in testing_list:
        last_folder_name = os.path.basename(feat_class_dir) + patient_name
        bw_dir = os.path.normpath(feat_class_dir + os.sep + os.pardir)
        bw_folder_name = os.path.basename(bw_dir)
        savefilename = os.path.basename(csvfile)
        save_feat_class_dir = feat_class_dir

    if not os.path.exists(save_feat_class_dir):
        os.makedirs(save_feat_class_dir)
    if "data_type" in args:
        data_type = args["data_type"]
        if data_type != None:
            if data_type == 'acceleration':
                np.savetxt(os.path.join(save_feat_class_dir, savefilename), 
                       np.array([timelist, xlist, ylist, zlist, accelerationlist]).transpose(), delimiter=',', header='time,x,y,z,acceleration') #save as csv file
            elif data_type == 'jerk':
                np.savetxt(os.path.join(save_feat_class_dir, savefilename), 
                       np.array([timelist, xlist, ylist, zlist, jerklist]).transpose(), delimiter=',', header='time,x,y,z,jerk') #save as csv file
            elif data_type == 'speed and acceleration':
                np.savetxt(os.path.join(save_feat_class_dir, savefilename), 
                       np.array([timelist, xlist, ylist, zlist, speedlist, accelerationlist]).transpose(), delimiter=',', header='time,x,y,z,speed,acceleration') #save as csv file
            elif data_type == 'speed and jerk':
                np.savetxt(os.path.join(save_feat_class_dir, savefilename), 
                       np.array([timelist, xlist, ylist, zlist, speedlist, jerklist]).transpose(), delimiter=',', header='time,x,y,z,speed,jerk') #save as csv file
            elif data_type == 'acceleration and jerk':
                np.savetxt(os.path.join(save_feat_class_dir, savefilename), 
                       np.array([timelist, xlist, ylist, zlist, accelerationlist, jerklist]).transpose(), delimiter=',', header='time,x,y,z,acceleration,jerk') #save as csv file
            elif data_type == 'all':
                np.savetxt(os.path.join(save_feat_class_dir, savefilename), 
                       np.array([timelist, xlist, ylist, zlist, speedlist, accelerationlist, jerklist]).transpose(), delimiter=',', header='time,x,y,z,speed,acceleration,jerk') #save as csv file
            else:
                np.savetxt(os.path.join(save_feat_class_dir, savefilename), 
                       np.array([timelist, xlist, ylist, zlist, speedlist]).transpose(), delimiter=',', header='time,x,y,z,speed') #save as csv file
    else:
        np.savetxt(os.path.join(save_feat_class_dir, savefilename), 
               np.array([timelist, xlist, ylist, zlist, speedlist]).transpose(), delimiter=',', header='time,x,y,z,speed') #save as csv file

def feature_extraction_without_time_3d_speed(csvfile, feat_class_dir, args, time_interval=1.0):
    use_header = True
    header_names = ["x", "y", "z"]
    if "use_header" in args:
        use_header = args["use_header"]
        if "header_names" in args:
            header_names = args["header_names"]
    if use_header:
        df = pd.read_csv(csvfile)
    else:
        df = pd.read_csv(csvfile, header=None, names=header_names)
    p_time = p_x = p_y = p_z = 0
    p_time2 = p_x2 = p_y2 = p_z2 = 0
    timelist = []
    speedlist = []
    for index, row in df.iterrows():
        time = index * time_interval
        x = row[0]
        y = row[1]
        z = row[2]
        if index > 0:
            timelist.append(time)
            dist = euc_dist_3d(x, y, z, p_x, p_y, p_z)
            speed = dist / (time - p_time)
            speedlist.append(speed)
        p_time2 = p_time
        p_x2 = p_x
        p_y2 = p_y
        p_z2 = p_z
        p_time = time
        p_x = x
        p_y = y
        p_z = z
    savefilename = os.path.basename(csvfile)
    np.savetxt(os.path.join(feat_class_dir, savefilename),
                   np.array([timelist, speedlist]).transpose(), delimiter=',', header='time,speed')
    
def euc_dist_3d(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

