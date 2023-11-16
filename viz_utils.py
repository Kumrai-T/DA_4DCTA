import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import os
import itertools
import numpy as np
import io_utils, attention
from IPython.display import HTML
import matplotlib.animation as animation
import math
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.text import Annotation
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 



def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)
    return arrow


setattr(Axes3D, 'arrow3D', _arrow3D)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    '''
    Confusion matrixを描く
    
    Parameters
    ----------
    cm : np.ndarray
        confusion matrix. sklearn.metrics.confusion_matrixで作成
    classes : list
        クラスのリスト
    normalize : Boolean
        ノーマライズしたConfusion matrixにするかどうか
    title : String
        図のタイトル
    cmap : cm.colormap
        色を塗るのに使うカラーマップ
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    

def plot_feature_time_color(scorelist, timelist, colorlist, ax, minv, maxv, linestyle = 'solid', linewidth=1.0):
    maxs=np.max(timelist,axis=0)
    mins=np.min(timelist,axis=0)
    for i in range(1,min(len(scorelist),len(colorlist))):
        line=plt.Line2D(timelist[i-1:i+1], scorelist[i-1:i+1], color=colorlist[i], linestyle = linestyle, linewidth= linewidth)
        ax.add_artist(line)
    ax.set_xlim(mins,maxs)
    margin= (maxv-minv)*0.05
    ax.set_ylim(minv-margin,maxv+margin)

def plot_att_feat_graph(data_file, att_file, ):
    feat_data_init = io_utils.get_one_data(data_file)
    
    if feat_data_init.shape[1] > 1:
        ylabel = ['speed', 'acceleration', 'jerk']
        for i in range(feat_data_init.shape[1]):
            feat_data = feat_data_init[:,i]
            att_data = io_utils.get_one_data(att_file)
            att_data = att_data[:,0]
            att_data = att_data[:len(feat_data)]
            minimum= np.min(att_data)
            maximum= np.max(att_data)
            colorlist = attention.score_to_color(att_data,minimum,maximum)
            fig=Figure(figsize=(8,4.5))
            ax2=fig.add_subplot(111)
            ax2.set_ylabel(ylabel[i])
            plot_feature_time_color(feat_data, [x for x in range(len(feat_data))],colorlist,ax2,np.min(feat_data),np.max(feat_data))
            display(fig)
    else:
        feat_data = feat_data_init.flatten()
        att_data = io_utils.get_one_data(att_file)
        att_data = att_data[:,0]
        att_data = att_data[:len(feat_data)]
        minimum= np.min(att_data)
        maximum= np.max(att_data)
        colorlist = attention.score_to_color(att_data,minimum,maximum)
        fig=Figure(figsize=(8,4.5))
        ax2=fig.add_subplot(111)
        ax2.set_ylabel("feature")
        plot_feature_time_color(feat_data, [x for x in range(len(feat_data))],colorlist,ax2,np.min(feat_data),np.max(feat_data))
        display(fig)

def plot_sum_att_feat_graph(data_file, att_file_list):
    feat_data_init = io_utils.get_one_data(data_file
    if feat_data_init.shape[1] > 1:
        ylabel = ['speed', 'acceleration', 'jerk']
        for i in range(feat_data_init.shape[1]):
            feat_data = feat_data_init[:,i]
            all_att_data = None
            for inx, att_file in enumerate(att_file_list):
                att_data = io_utils.get_one_data(att_file)
                att_data = att_data[:,0]
                att_data = att_data[:len(feat_data)]
                if inx == 0:
                    all_att_data = att_data.copy().reshape((-1,1))
                else:
                    all_att_data = np.concatenate((all_att_data, att_data.copy().reshape((-1,1))), axis=1)
            sum_att_data = np.sum(all_att_data, axis=1)
            sum_att_data = np.reshape(sum_att_data, (-1,))
            minimum= np.min(sum_att_data)
            maximum= np.max(sum_att_data)
            colorlist = attention.score_to_color(sum_att_data,minimum,maximum)
            fig=Figure(figsize=(8,4.5))
            ax2=fig.add_subplot(111)
            ax2.set_ylabel(ylabel[i])
            plot_feature_time_color(feat_data, [x for x in range(len(feat_data))],colorlist,ax2,np.min(feat_data),np.max(feat_data))
            display(fig)
    else:
        ylabel = ['speed']
        for i in range(feat_data_init.shape[1]):
            feat_data = feat_data_init[:,i]
            all_att_data = None
            for inx, att_file in enumerate(att_file_list):
                att_data = io_utils.get_one_data(att_file)
                att_data = att_data[:,0]
                att_data = att_data[:len(feat_data)]
                if inx == 0:
                    all_att_data = att_data.copy().reshape((-1,1))
                else:
                    all_att_data = np.concatenate((all_att_data, att_data.copy().reshape((-1,1))), axis=1)
            sum_att_data = np.sum(all_att_data, axis=1)
            sum_att_data = np.reshape(sum_att_data, (-1,))
            minimum= np.min(sum_att_data)
            maximum= np.max(sum_att_data)
            colorlist = attention.score_to_color(sum_att_data,minimum,maximum)
            fig=Figure(figsize=(8,4.5))
            ax2=fig.add_subplot(111)
            ax2.set_ylabel(ylabel[i])
            plot_feature_time_color(feat_data, [x for x in range(len(feat_data))],colorlist,ax2,np.min(feat_data),np.max(feat_data))
            display(fig)

def plot_avg_att_feat_graph(data_file, att_file_list):
    feat_data_init = io_utils.get_one_data(data_file)
    if feat_data_init.shape[1] > 1:
        ylabel = ['speed', 'acceleration', 'jerk']
        for i in range(feat_data_init.shape[1]):
            feat_data = feat_data_init[:,i]
            all_att_data = None
            for inx, att_file in enumerate(att_file_list):
                att_data = io_utils.get_one_data(att_file)
                att_data = att_data[:,0]
                att_data = att_data[:len(feat_data)]
                if inx == 0:
                    all_att_data = att_data.copy().reshape((-1,1))
                else:
                    all_att_data = np.concatenate((all_att_data, att_data.copy().reshape((-1,1))), axis=1)
            avg_att_data = np.mean(all_att_data, axis=1)
            avg_att_data = np.reshape(avg_att_data, (-1,))
            minimum= np.min(avg_att_data)
            maximum= np.max(avg_att_data)
            colorlist = attention.score_to_color(avg_att_data,minimum,maximum)
            fig=Figure(figsize=(8,4.5))
            ax2=fig.add_subplot(111)
            ax2.set_ylabel(ylabel[i])
            plot_feature_time_color(feat_data, [x for x in range(len(feat_data))],colorlist,ax2,np.min(feat_data),np.max(feat_data))
            display(fig)
    else:
        ylabel = ['speed']
        for i in range(feat_data_init.shape[1]):
            feat_data = feat_data_init[:,i]
            all_att_data = None
            for inx, att_file in enumerate(att_file_list):
                att_data = io_utils.get_one_data(att_file)
                att_data = att_data[:,0]
                att_data = att_data[:len(feat_data)]
                if inx == 0:
                    all_att_data = att_data.copy().reshape((-1,1))
                else:
                    all_att_data = np.concatenate((all_att_data, att_data.copy().reshape((-1,1))), axis=1)
            avg_att_data = np.mean(all_att_data, axis=1)
            avg_att_data = np.reshape(avg_att_data, (-1,))
            minimum= np.min(avg_att_data)
            maximum= np.max(avg_att_data)
            colorlist = attention.score_to_color(avg_att_data,minimum,maximum)
            fig=Figure(figsize=(8,4.5))
            ax2=fig.add_subplot(111)
            ax2.set_ylabel(ylabel[i])
            plot_feature_time_color(feat_data, [x for x in range(len(feat_data))],colorlist,ax2,np.min(feat_data),np.max(feat_data))
            display(fig)
    
def plot_all_att_feat_graph(data_file, layers, att_file_template):
    feat_data_init = io_utils.get_one_data(data_file)
    if feat_data_init.shape[1] > 1:
        for i in range(feat_data_init.shape[1]):
            feat_data = feat_data_init[:,i]
            att_data = None
            for layer in layers:
                att_file = att_file_template.replace("layer_name", layer)
                tmp_att_data = io_utils.get_one_data(att_file)
                tmp_att_data = tmp_att_data[:,0]
                tmp_att_data = tmp_att_data[:len(feat_data)]
                att_data = tmp_att_data if att_data is None else att_data + tmp_att_data
            att_data = att_data / len(layers)
            minimum= np.min(att_data)
            maximum= np.max(att_data)
            colorlist = attention.score_to_color(att_data,minimum,maximum)
            fig=Figure(figsize=(8,4.5))
            ax2=fig.add_subplot(111)
            ax2.set_ylabel("feature")
            plot_feature_time_color(feat_data, [x for x in range(len(att_data))],colorlist,ax2,np.min(feat_data),np.max(feat_data))
            display(fig)
    else:
        feat_data = feat_data_init.flatten()
        att_data = None
        for layer in layers:
            att_file = att_file_template.replace("layer_name", layer)
            tmp_att_data = io_utils.get_one_data(att_file)
            tmp_att_data = tmp_att_data[:,0]
            tmp_att_data = tmp_att_data[:len(feat_data)]
            att_data = tmp_att_data if att_data is None else att_data + tmp_att_data
        att_data = att_data / len(layers)
        minimum= np.min(att_data)
        maximum= np.max(att_data)
        colorlist = attention.score_to_color(att_data,minimum,maximum)
        fig=Figure(figsize=(8,4.5))
        ax2=fig.add_subplot(111)
        ax2.set_ylabel("feature")
        plot_feature_time_color(feat_data, [x for x in range(len(att_data))],colorlist,ax2,np.min(feat_data),np.max(feat_data))
        display(fig)
    
    
def plot_all_att_3D(data_file, layers, att_file_template):
    feat_data = io_utils.get_one_data(data_file)
    att_data = None
    for layer in layers:
        att_file = att_file_template.replace("layer_name", layer)
        tmp_att_data = io_utils.get_one_data(att_file)
        tmp_att_data = tmp_att_data[:,0]
        att_data = tmp_att_data if att_data is None else att_data + tmp_att_data
    att_data = att_data / len(layers)
    minimum= np.min(att_data)
    maximum= np.max(att_data)
    colorlist = attention.score_to_color(att_data,minimum,maximum)
    feat_data = feat_data[:len(colorlist)]
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(data_file.split('/')[-1], size = 20)
    ax.set_xlabel("x", size = 14)
    ax.set_ylabel("y", size = 14)
    ax.set_zlabel("z", size = 14)
    x = feat_data[:,0]
    y = feat_data[:,1]
    z = feat_data[:,2]
    ax.scatter(x, y, z, s = 40, c = colorlist)
    ax.view_init(elev=45, azim=45)
    plt.show()

def plot_avg_all_att_3D(data_file, layers, att_file_list):
    feat_data_init = io_utils.get_one_data(data_file, include_time=True)
    print(data_file)
    print(feat_data_init.shape)
    if feat_data_init.shape[1] > 1:
        ylabel = ['speed', 'acceleration', 'jerk']
        for i in range(feat_data_init.shape[1]-1):
            feat_data = feat_data_init[:,i+1]
            all_att_data = None
            for inx, att_file in enumerate(att_file_list):
                att_data = io_utils.get_one_data(att_file)
                att_data = att_data[:,0]
                att_data = att_data[:len(feat_data)]
                if inx == 0:
                    all_att_data = att_data.copy().reshape((-1,1))
                else:
                    all_att_data = np.concatenate((all_att_data, att_data.copy().reshape((-1,1))), axis=1)
            avg_att_data = np.mean(all_att_data, axis=1)
            avg_att_data = np.reshape(avg_att_data, (-1,))
    avg_att_data = avg_att_data / len(layers)
    minimum= np.min(avg_att_data)
    maximum= np.max(avg_att_data)
    colorlist, colorlist_sorted = attention.score_to_color(avg_att_data,minimum,maximum)
    feat_data_init = feat_data_init[:len(colorlist)]
    fig = plt.figure(figsize = (10, 16))
    ax1 = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212, projection='3d')
    ax1.set_title(data_file.split('/')[-1], size = 20)
    ax1.set_xlabel("x", size = 14)
    ax1.set_ylabel("y", size = 14)
    ax1.set_zlabel("z", size = 14)
    ax2.set_xlabel("x", size = 14)
    ax2.set_ylabel("y", size = 14)
    ax2.set_zlabel("z", size = 14)
    time_t = feat_data_init[:,0]
    x = feat_data_init[:,1]
    y = feat_data_init[:,2]
    z = feat_data_init[:,3]

    dist1 = euclidean_norm(feat_data_init[:,1:])
    dist1.append(np.linalg.norm(feat_data_init[:,1:][0]-feat_data_init[:,1:][-1]))
    idx_outlier = [i for i in range(len(dist1)) if dist1[i] >= 0.2]

    if len(idx_outlier) > 0:
        new_feat_data_init = []
        for idx in idx_outlier:
            if idx+1 < 100:
                new_feat_data_init.append(feat_data_init[idx])
                new_feat_data_init.append(feat_data_init[idx+1])

        new_feat_data_init = np.array(new_feat_data_init)
        print(new_feat_data_init.shape)
        new2_x = new_feat_data_init[:,1]
        new2_y = new_feat_data_init[:,2]
        new2_z = new_feat_data_init[:,3]

    ax1.plot3D(x, y, z, 'black', alpha=0.5)
    c_list = np.array(colorlist)
    c_list_sorted = np.array(colorlist_sorted)
    cm = LinearSegmentedColormap.from_list('defcol', c_list_sorted)
    p1 = ax1.scatter(x, y, z, s = 20, c = colorlist)
    fig.colorbar(ScalarMappable(cmap=cm, norm=plt.Normalize(0, 1)), ax=ax1)
    if len(idx_outlier) > 0:
        ax1.scatter(new2_x, new2_y, new2_z, s = 40, c='black')
    ax1.view_init(elev=45, azim=45)
    ax2.view_init(elev=45, azim=45)

    speedlist, distlist = [], []
    p_time = p_x = p_y = p_z = 0
    for idx_n in range(len(time_t)):
        time = time_t[idx_n]
        x_i = x[idx_n]
        y_i = y[idx_n]
        z_i = z[idx_n]
        if idx_n > 0:
            dist_i = euc_dist_3d(x_i, y_i, z_i, p_x, p_y, p_z)
            speed_i = dist_i / (time - p_time)
            speedlist.append(speed_i)
            if dist_i <= 0.1:
                distlist.append(dist_i)
        else:
            speed_i, dist_i = 0, 0
            speedlist.append(0)
            distlist.append(0)
        p_time = time
        p_x = x_i
        p_y = y_i
        p_z = z_i

    sudden_speedlist = [i for i in range(len(speedlist)) if speedlist[i] >= 0.1]

    sudden_speedlist2 = []
    for j in range(1, len(speedlist)):
        if j-2 in sudden_speedlist2:
            continue
        if (speedlist[j] >= 0.1 and speedlist[j-1] >= 0.1
            sudden_speedlist2.append(j-1)
            j += 2
        elif speedlist[j] >= 0.042 and speedlist[j-1] >= 0.042:
            sudden_speedlist2.append(j-1)
            j += 2
        else:
            pass

    x_new = x.copy()
    y_new = y.copy()
    z_new = z.copy()
    for i, idx in enumerate(sudden_speedlist2): 
        if idx+1<100:
            x_new[idx] = (x[idx-1]+x[idx+1])/2
            y_new[idx] = (y[idx-1]+y[idx+1])/2
            z_new[idx] = (z[idx-1]+z[idx+1])/2

    ax2.plot3D(x_new, y_new, z_new, 'red', alpha=0.5)
    c_list = np.array(colorlist)
    c_list_sorted = np.array(colorlist_sorted)
    cm = LinearSegmentedColormap.from_list('defcol', c_list_sorted)
    p2 = ax2.scatter(x_new, y_new, z_new, s = 40, c = colorlist)
    fig.colorbar(ScalarMappable(cmap=cm, norm=plt.Normalize(0, 1)), ax=ax2)

    speedlist = []
    p_time = p_x = p_y = p_z = 0
    for idx_n in range(len(time_t)):
        time = time_t[idx_n]
        x_i = x_new[idx_n]
        y_i = y_new[idx_n]
        z_i = z_new[idx_n]
        if idx_n > 0:
            dist_i = euc_dist_3d(x_i, y_i, z_i, p_x, p_y, p_z)
            speed_i = dist_i / (time - p_time)
            speedlist.append(speed_i)
        else:
            speed_i = 0
            speedlist.append(0)
        p_time = time
        p_x = x_i
        p_y = y_i
        p_z = z_i

    savepath = './Pictures/'
    
    if 'blue' in data_file and 'skyblue' not in data_file:
        fig = plt.figure(figsize = (10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("3D trajectory\nof TW", size = 20)
        ax.plot3D(x_new, y_new, z_new, 'blue', alpha=0.2)

        u = np.diff(x_new)
        v = np.diff(y_new)
        w = np.diff(z_new)
        pos_x = x_new[:-1] + u/2
        pos_y = y_new[:-1] + v/2
        pos_z = z_new[:-1] + w/2
        norm = np.sqrt(u**2+v**2+w**2) * 30
        newpos_x = np.array([pos_x[i] for i in range(len(pos_x)) if i % 8 == 1])
        newpos_y = np.array([pos_y[i] for i in range(len(pos_y)) if i % 8 == 1])
        newpos_z = np.array([pos_z[i] for i in range(len(pos_z)) if i % 8 == 1])
        new_norm = np.array([norm[i] for i in range(len(norm)) if i % 8 == 1])
        new_u = np.array([u[i] for i in range(len(u)) if i % 8 == 1])
        new_v = np.array([v[i] for i in range(len(v)) if i % 8 == 1])
        new_w = np.array([w[i] for i in range(len(w)) if i % 8 == 1])

        ax.quiver(newpos_x+0.05, newpos_y, newpos_z, new_u/new_norm, new_v/new_norm, new_w/new_norm, zorder=3, pivot="middle", color='black', alpha=0.5, arrow_length_ratio=0.5)
        c_list = np.array(colorlist)
        c_list_sorted = np.array(colorlist_sorted)
        cm = LinearSegmentedColormap.from_list('defcol', c_list_sorted)
        p_save = ax.scatter(x_new, y_new, z_new, s = 70, c = c_list)
        fig.colorbar(ScalarMappable(cmap=cm, norm=plt.Normalize(0, 1)), ax=ax)
        dir_name = savepath + '3D_trajectory_of_TW_'+ data_file.split('/')[-1][:-4] + '.png'
    elif 'red' in data_file:
        fig = plt.figure(figsize = (10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("3D trajectory\nof HR", size = 20)
        ax.plot3D(x_new, y_new, z_new, 'red', alpha=0.2)
        u = np.diff(x_new)
        v = np.diff(y_new)
        w = np.diff(z_new)
        pos_x = x_new[:-1] + u/2
        pos_y = y_new[:-1] + v/2
        pos_z = z_new[:-1] + w/2
        norm = np.sqrt(u**2+v**2+w**2) * 30
        newpos_x = np.array([pos_x[i] for i in range(len(pos_x)) if i % 8 == 1])
        newpos_y = np.array([pos_y[i] for i in range(len(pos_y)) if i % 8 == 1])
        newpos_z = np.array([pos_z[i] for i in range(len(pos_z)) if i % 8 == 1])
        new_norm = np.array([norm[i] for i in range(len(norm)) if i % 8 == 1])
        new_u = np.array([u[i] for i in range(len(u)) if i % 8 == 1])
        new_v = np.array([v[i] for i in range(len(v)) if i % 8 == 1])
        new_w = np.array([w[i] for i in range(len(w)) if i % 8 == 1])
        ax.quiver(newpos_x+0.05, newpos_y, newpos_z, new_u/new_norm, new_v/new_norm, new_w/new_norm, zorder=3, pivot="middle", color='black', alpha=0.5, arrow_length_ratio=0.5)
        c_list = np.array(colorlist)
        c_list_sorted = np.array(colorlist_sorted)
        cm = LinearSegmentedColormap.from_list('defcol', c_list_sorted)
        p_save = ax.scatter(x_new, y_new, z_new, s = 70, c = c_list)
        fig.colorbar(ScalarMappable(cmap=cm, norm=plt.Normalize(0, 1)), ax=ax)
        dir_name = savepath + '3D_trajectory_of_HR_'+ data_file.split('/')[-1][:-4] + '.png'
    else:
        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(x_new, y_new, z_new, 'black', alpha=0.2)
        u = np.diff(x_new)
        v = np.diff(y_new)
        w = np.diff(z_new)
        pos_x = x_new[:-1] + u/2
        pos_y = y_new[:-1] + v/2
        pos_z = z_new[:-1] + w/2
        norm = np.sqrt(u**2+v**2+w**2) * 30
        newpos_x = np.array([pos_x[i] for i in range(len(pos_x)) if i % 9 == 1])
        newpos_y = np.array([pos_y[i] for i in range(len(pos_y)) if i % 9 == 1])
        newpos_z = np.array([pos_z[i] for i in range(len(pos_z)) if i % 9 == 1])
        new_norm = np.array([norm[i] for i in range(len(norm)) if i % 9 == 1])
        new_u = np.array([u[i] for i in range(len(u)) if i % 9 == 1])
        new_v = np.array([v[i] for i in range(len(v)) if i % 9 == 1])
        new_w = np.array([w[i] for i in range(len(w)) if i % 9 == 1])
        ax.quiver(newpos_x+0.05, newpos_y, newpos_z, new_u/new_norm, new_v/new_norm, new_w/new_norm, zorder=3, pivot="middle", color='blue', alpha=0.7, arrow_length_ratio=0.7)
        ax.scatter(x_new, y_new, z_new, s = 70, c = 'red')
        dir_name = savepath + '3D_trajectory_'+ data_file.split('/')[-1][:-4] + '.png'
    ax.set_xlabel("x", size = 18)
    ax.set_ylabel("y", size = 18)
    ax.set_zlabel("z", size = 18)
    ax.view_init(elev=25, azim=-130)
    
    fig.tight_layout()
    plt.savefig(dir_name, bbox_inches='tight', dpi=300)

def plot_avg_all_att_3D_all_traj(data_file, layers, att_file_list):
    feat_data_init = io_utils.get_one_data(data_file, include_time=True)
    print(data_file)
    #feat_data = feat_data.flatten()
    if feat_data_init.shape[1] > 1:
        ylabel = ['speed', 'acceleration', 'jerk']
        for i in range(feat_data_init.shape[1]-1):
            feat_data = feat_data_init[:,i+1]
            all_att_data = None
            for inx, att_file in enumerate(att_file_list):
                #print(att_file)
                att_data = io_utils.get_one_data(att_file)
                att_data = att_data[:,0]
                att_data = att_data[:len(feat_data)]
                if inx == 0:
                    all_att_data = att_data.copy().reshape((-1,1))
                else:
                    all_att_data = np.concatenate((all_att_data, att_data.copy().reshape((-1,1))), axis=1)
            avg_att_data = np.mean(all_att_data, axis=1)
            avg_att_data = np.reshape(avg_att_data, (-1,))
    avg_att_data = avg_att_data / len(layers)
    minimum= np.min(avg_att_data)
    maximum= np.max(avg_att_data)
    colorlist, colorlist_sorted = attention.score_to_color(avg_att_data,minimum,maximum)
    feat_data_init = feat_data_init[:len(colorlist)]
#         fig = plt.figure(figsize = (10, 16))
#         ax1 = fig.add_subplot(211, projection='3d')
#         ax2 = fig.add_subplot(212, projection='3d')
#         ax1.set_title(data_file.split('/')[-1], size = 20)
#         ax1.set_xlabel("x", size = 14)
#         ax1.set_ylabel("y", size = 14)
#         ax1.set_zlabel("z", size = 14)
#         ax2.set_xlabel("x", size = 14)
#         ax2.set_ylabel("y", size = 14)
#         ax2.set_zlabel("z", size = 14)
    time_t = feat_data_init[:,0]
    x = feat_data_init[:,1]
    y = feat_data_init[:,2]
    z = feat_data_init[:,3]

    dist1 = euclidean_norm(feat_data_init[:,1:])
    dist1.append(np.linalg.norm(feat_data_init[:,1:][0]-feat_data_init[:,1:][-1]))
    idx_outlier = [i for i in range(len(dist1)) if dist1[i] >= 0.2]

    if len(idx_outlier) > 0:
        new_feat_data_init = []
        for idx in idx_outlier:
            if idx+1 < 100:
                new_feat_data_init.append(feat_data_init[idx])
                new_feat_data_init.append(feat_data_init[idx+1])

        new_feat_data_init = np.array(new_feat_data_init)
        print(new_feat_data_init.shape)
        new2_x = new_feat_data_init[:,1]
        new2_y = new_feat_data_init[:,2]
        new2_z = new_feat_data_init[:,3]

    speedlist, distlist = [], []
    p_time = p_x = p_y = p_z = 0
    for idx_n in range(len(time_t)):
        time = time_t[idx_n]
        x_i = x[idx_n]
        y_i = y[idx_n]
        z_i = z[idx_n]
        if idx_n > 0:
            dist_i = euc_dist_3d(x_i, y_i, z_i, p_x, p_y, p_z)
            speed_i = dist_i / (time - p_time)
            speedlist.append(speed_i)
            if dist_i <= 0.1:
                distlist.append(dist_i)
        else:
            speed_i, dist_i = 0, 0
            speedlist.append(0)
            distlist.append(0)
        p_time = time
        p_x = x_i
        p_y = y_i
        p_z = z_i

    sudden_speedlist = [i for i in range(len(speedlist)) if speedlist[i] >= 0.1]

    sudden_speedlist2 = []
    for j in range(1, len(speedlist)):
        if j-2 in sudden_speedlist2:
            continue
        if speedlist[j] >= 0.1 and speedlist[j-1] >= 0.1:
            sudden_speedlist2.append(j-1)
            j += 2
        elif speedlist[j] >= 0.042 and speedlist[j-1] >= 0.042:
            sudden_speedlist2.append(j-1)
            j += 2
        else:
            pass

    x_new = x.copy()
    y_new = y.copy()
    z_new = z.copy()
    for i, idx in enumerate(sudden_speedlist2): 
        if idx+1<100:
            x_new[idx] = (x[idx-1]+x[idx+1])/2
            y_new[idx] = (y[idx-1]+y[idx+1])/2
            z_new[idx] = (z[idx-1]+z[idx+1])/2

    speedlist = []
    p_time = p_x = p_y = p_z = 0
    for idx_n in range(len(time_t)):
        time = time_t[idx_n]
        x_i = x_new[idx_n]
        y_i = y_new[idx_n]
        z_i = z_new[idx_n]
        if idx_n > 0:
            dist_i = euc_dist_3d(x_i, y_i, z_i, p_x, p_y, p_z)
            speed_i = dist_i / (time - p_time)
            speedlist.append(speed_i)
        else:
            speed_i = 0
            speedlist.append(0)
        p_time = time
        p_x = x_i
        p_y = y_i
        p_z = z_i


    savepath = './Pictures/'

    if 'blue' in data_file and 'skyblue' not in data_file:
        fig = plt.figure(figsize = (10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("3D trajectory\nof TW", size = 20)
        ax.plot3D(x_new, y_new, z_new, 'blue', alpha=0.2)

        u = np.diff(x_new)
        v = np.diff(y_new)
        w = np.diff(z_new)
        pos_x = x_new[:-1] + u/2
        pos_y = y_new[:-1] + v/2
        pos_z = z_new[:-1] + w/2
        norm = np.sqrt(u**2+v**2+w**2) * 30
        newpos_x = np.array([pos_x[i] for i in range(len(pos_x)) if i % 8 == 1])
        newpos_y = np.array([pos_y[i] for i in range(len(pos_y)) if i % 8 == 1])
        newpos_z = np.array([pos_z[i] for i in range(len(pos_z)) if i % 8 == 1])
        new_norm = np.array([norm[i] for i in range(len(norm)) if i % 8 == 1])
        new_u = np.array([u[i] for i in range(len(u)) if i % 8 == 1])
        new_v = np.array([v[i] for i in range(len(v)) if i % 8 == 1])
        new_w = np.array([w[i] for i in range(len(w)) if i % 8 == 1])

        ax.quiver(newpos_x+0.05, newpos_y, newpos_z, new_u/new_norm, new_v/new_norm, new_w/new_norm, zorder=3, pivot="middle", color='black', alpha=0.5, arrow_length_ratio=0.5)
        c_list = np.array(colorlist)
        c_list_sorted = np.array(colorlist_sorted)
        cm = LinearSegmentedColormap.from_list('defcol', c_list_sorted)
        p_save = ax.scatter(x_new, y_new, z_new, s = 70, c = c_list)
        fig.colorbar(ScalarMappable(cmap=cm, norm=plt.Normalize(0, 1)), ax=ax)
        dir_name = savepath + '3D_trajectory_of_TW_'+ data_file.split('/')[-1][:-4] + '.png'
    elif 'red' in data_file:
        fig = plt.figure(figsize = (10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("3D trajectory\nof HR", size = 20)
        ax.plot3D(x_new, y_new, z_new, 'red', alpha=0.2)
        u = np.diff(x_new)
        v = np.diff(y_new)
        w = np.diff(z_new)
        pos_x = x_new[:-1] + u/2
        pos_y = y_new[:-1] + v/2
        pos_z = z_new[:-1] + w/2
        norm = np.sqrt(u**2+v**2+w**2) * 30
        newpos_x = np.array([pos_x[i] for i in range(len(pos_x)) if i % 8 == 1])
        newpos_y = np.array([pos_y[i] for i in range(len(pos_y)) if i % 8 == 1])
        newpos_z = np.array([pos_z[i] for i in range(len(pos_z)) if i % 8 == 1])
        new_norm = np.array([norm[i] for i in range(len(norm)) if i % 8 == 1])
        new_u = np.array([u[i] for i in range(len(u)) if i % 8 == 1])
        new_v = np.array([v[i] for i in range(len(v)) if i % 8 == 1])
        new_w = np.array([w[i] for i in range(len(w)) if i % 8 == 1])
        ax.quiver(newpos_x+0.05, newpos_y, newpos_z, new_u/new_norm, new_v/new_norm, new_w/new_norm, zorder=3, pivot="middle", color='black', alpha=0.5, arrow_length_ratio=0.5)
        c_list = np.array(colorlist)
        c_list_sorted = np.array(colorlist_sorted)
        cm = LinearSegmentedColormap.from_list('defcol', c_list_sorted)
        p_save = ax.scatter(x_new, y_new, z_new, s = 70, c = c_list)
        fig.colorbar(ScalarMappable(cmap=cm, norm=plt.Normalize(0, 1)), ax=ax)
        dir_name = savepath + '3D_trajectory_of_HR_'+ data_file.split('/')[-1][:-4] + '.png'
    else:
        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(x_new, y_new, z_new, 'black', alpha=0.2)
        u = np.diff(x_new)
        v = np.diff(y_new)
        w = np.diff(z_new)
        pos_x = x_new[:-1] + u/2
        pos_y = y_new[:-1] + v/2
        pos_z = z_new[:-1] + w/2
        norm = np.sqrt(u**2+v**2+w**2) * 30
        newpos_x = np.array([pos_x[i] for i in range(len(pos_x)) if i % 9 == 1])
        newpos_y = np.array([pos_y[i] for i in range(len(pos_y)) if i % 9 == 1])
        newpos_z = np.array([pos_z[i] for i in range(len(pos_z)) if i % 9 == 1])
        new_norm = np.array([norm[i] for i in range(len(norm)) if i % 9 == 1])
        new_u = np.array([u[i] for i in range(len(u)) if i % 9 == 1])
        new_v = np.array([v[i] for i in range(len(v)) if i % 9 == 1])
        new_w = np.array([w[i] for i in range(len(w)) if i % 9 == 1])
        ax.quiver(newpos_x+0.05, newpos_y, newpos_z, new_u/new_norm, new_v/new_norm, new_w/new_norm, zorder=3, pivot="middle", color='blue', alpha=0.7, arrow_length_ratio=0.7)
        ax.scatter(x_new, y_new, z_new, s = 70, c = 'red')
        dir_name = savepath + '3D_trajectory_'+ data_file.split('/')[-1][:-4] + '.png'
    #ax1.set_title(data_file.split('/')[-1], size = 20)
    ax.set_xlabel("x", size = 18)
    ax.set_ylabel("y", size = 18)
    ax.set_zlabel("z", size = 18)
    ax.view_init(elev=20, azim=-155)

    fig.tight_layout()
    plt.savefig(dir_name, bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()

def point_displacement(point, vec, disp):
    unit_vec = vec / np.linalg.norm(vec)
    return point + disp * unit_vec

def euc_dist_3d(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

# numpy.median is rather slow, let's build our own instead
def median(x):
    m,n = x.shape
    middle = np.arange((m-1)>>1,(m>>1)+1)
    x = np.partition(x,middle,axis=0)
    return x[middle].mean(axis=0)

# main function
def remove_outliers(data,thresh=2.0):           
    m = median(data)                            
    s = np.abs(data-m)                          
    return data[(s<median(s)*thresh).all(axis=1)]

def find_xy(p1, p2, z):

    x1, y1, z1 = p1
    x2, y2, z2 = p2
    if z2 < z1:
        return find_xy(p2, p1, z)
    
    x = np.interp(z, (z1, z2), (x1, x2))
    y = np.interp(z, (z1, z2), (y1, y2))

    return x, y

def moving_avg(arr):
    x=0.5  # smoothening factor
  
    i = 1
    # Initialize an empty list to
    # store exponential moving averages
    moving_averages = []
  
    # Insert first exponential average in the list
    moving_averages.append(arr[0])
  
    # Loop through the array elements
    while i < len(arr):
  
        # Calculate the exponential
        # average by using the formula
        window_average = round((x*arr[i])+(1-x)*moving_averages[-1], 2)
      
        # Store the cumulative average
        # of current window in moving average list
        moving_averages.append(window_average)
      
        # Shift window to right by one position
        i += 1
  
    return np.array(moving_averages)

def euclidean_norm(narray):
    dist = []
    for i in range(len(narray)-1):
        dist.append(np.linalg.norm(narray[i]-narray[i+1]))
    return dist

def euclidean_dist(narray):
    dist = []
    for i in range(len(narray)-1):
        dist.append(np.sqrt(np.sum(np.square(narray[i]-narray[i+1]))))
    return dist

def plot_sum_all_att_3D(data_file, layers, att_file_list):
    feat_data_init = io_utils.get_one_data(data_file, include_time=True)
    print(data_file)
    print(feat_data_init.shape)
    #feat_data = feat_data.flatten()
    if feat_data_init.shape[1] > 1:
        ylabel = ['speed', 'acceleration', 'jerk']
        for i in range(feat_data_init.shape[1]-1):
            feat_data = feat_data_init[:,i+1]
            all_att_data = None
            for inx, att_file in enumerate(att_file_list):
                #print(att_file)
                att_data = io_utils.get_one_data(att_file)
                att_data = att_data[:,0]
                att_data = att_data[:len(feat_data)]
                if inx == 0:
                    all_att_data = att_data.copy().reshape((-1,1))
                else:
                    all_att_data = np.concatenate((all_att_data, att_data.copy().reshape((-1,1))), axis=1)
                #print("     ", att_data.shape, all_att_data.shape)
            avg_att_data = np.sum(all_att_data, axis=1)
            avg_att_data = np.reshape(avg_att_data, (-1,))
#            print(avg_att_data.shape, avg_att_data[:2])
    avg_att_data = avg_att_data / len(layers)
    minimum= np.min(avg_att_data)
    maximum= np.max(avg_att_data)
    colorlist = attention.score_to_color(avg_att_data,minimum,maximum)
    feat_data_init = feat_data_init[:len(colorlist)]
    fig = plt.figure(figsize = (10, 16))
    ax1 = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212, projection='3d')
    ax1.set_title(data_file.split('/')[-1], size = 20)
    ax1.set_xlabel("x", size = 14)
    ax1.set_ylabel("y", size = 14)
    ax1.set_zlabel("z", size = 14)
    #print(len(colorlist))
    #print(feat_data_init)
    time_t = feat_data_init[:,0]
    x = feat_data_init[:,1]
    y = feat_data_init[:,2]
    z = feat_data_init[:,3]

    dist1 = euclidean_norm(feat_data_init[:,1:])
    dist1.append(np.linalg.norm(feat_data_init[:,1:][0]-feat_data_init[:,1:][-1]))
    #print(len(dist1), dist1)
    idx_outlier = [i for i in range(len(dist1)) if dist1[i] >= 0.2]
    #print(len(idx_outlier), idx_outlier)

    if len(idx_outlier) > 0:
        new_feat_data_init = []
        for idx in idx_outlier:
            new_feat_data_init.append(feat_data_init[idx])
            new_feat_data_init.append(feat_data_init[idx+1])

        new_feat_data_init = np.array(new_feat_data_init)
        print(new_feat_data_init.shape)
        new2_x = new_feat_data_init[:,1]
        new2_y = new_feat_data_init[:,2]
        new2_z = new_feat_data_init[:,3]

    ax1.plot3D(x, y, z, 'black', alpha=0.5)
    p1 = ax1.scatter(x, y, z, s = 20, c = colorlist)#, c='blue', alpha=0.2)#, c = colorlist)
    fig.colorbar(p1, ax=ax1)
    if len(idx_outlier) > 0:
        ax1.scatter(new2_x, new2_y, new2_z, s = 40, c='black')
    ax1.view_init(elev=45, azim=45)
    ax2.view_init(elev=45, azim=45)

    speedlist, distlist = [], []
    p_time = p_x = p_y = p_z = 0
    for idx_n in range(len(time_t)):
        time = time_t[idx_n]
        x_i = x[idx_n]
        y_i = y[idx_n]
        z_i = z[idx_n]
        if idx_n > 0:
            dist_i = euc_dist_3d(x_i, y_i, z_i, p_x, p_y, p_z)
            speed_i = dist_i / (time - p_time)
            speedlist.append(speed_i)
            if dist_i <= 0.1:
                distlist.append(dist_i)
        else:
            speed_i, dist_i = 0, 0
            speedlist.append(0)
            distlist.append(0)
        p_time = time
        p_x = x_i
        p_y = y_i
        p_z = z_i

    sudden_speedlist = [i for i in range(len(speedlist)) if speedlist[i] >= 0.1]
    sudden_speedlist2 = []
    for j in range(1, len(speedlist)):
        if j-2 in sudden_speedlist2:
            continue
        if speedlist[j] >= 0.1 and speedlist[j-1] >= 0.1:
            sudden_speedlist2.append(j-1)
            j += 2
        elif speedlist[j] >= 0.042 and speedlist[j-1] >= 0.042:
            sudden_speedlist2.append(j-1)
            j += 2
        else:
            pass

    x_new = x.copy()
    y_new = y.copy()
    z_new = z.copy()
    for i, idx in enumerate(sudden_speedlist2): 
        if idx+1<100:
            x_new[idx] = (x[idx-1]+x[idx+1])/2
            y_new[idx] = (y[idx-1]+y[idx+1])/2
            z_new[idx] = (z[idx-1]+z[idx+1])/2
    ax2.plot3D(x_new, y_new, z_new, 'red', alpha=0.5)
    p2 = ax2.scatter(x_new, y_new, z_new, s = 40, c = colorlist)
    fig.colorbar(p2, ax=ax2)

    speedlist = []
    p_time = p_x = p_y = p_z = 0
    for idx_n in range(len(time_t)):
        time = time_t[idx_n]
        x_i = x_new[idx_n]
        y_i = y_new[idx_n]
        z_i = z_new[idx_n]
        if idx_n > 0:
            dist_i = euc_dist_3d(x_i, y_i, z_i, p_x, p_y, p_z)
            speed_i = dist_i / (time - p_time)
            speedlist.append(speed_i)
        else:
            speed_i = 0
            speedlist.append(0)
        p_time = time
        p_x = x_i
        p_y = y_i
        p_z = z_i

    plt.show()
    
class Ani3DRotView:
    def __init__(self, data_file, layers, att_file_template):
        feat_data = io_utils.get_one_data(data_file)#, include_time=True)
        att_data = None
        for layer in layers:
            att_file = att_file_template.replace("layer_name", layer)
            tmp_att_data = io_utils.get_one_data(att_file)
            tmp_att_data = tmp_att_data[:,0]
            att_data = tmp_att_data if att_data is None else att_data + tmp_att_data
        att_data = att_data / len(layers)
        minimum= np.min(att_data)
        maximum= np.max(att_data)
        att_data = att_data[:feat_data.shape[0]]
        colorlist = attention.score_to_color(att_data,minimum,maximum)
        feat_data = feat_data[:len(colorlist)]
        self.fig = plt.figure(figsize = (8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title(data_file.split('/')[-1], size = 20)
        self.ax.set_xlabel("x", size = 14)
        self.ax.set_ylabel("y", size = 14)
        self.ax.set_zlabel("z", size = 14)
        self.x = feat_data[:,0]
        self.y = feat_data[:,1]
        self.z = feat_data[:,2]
        self.colorlist = colorlist
        self.title=data_file.split('/')[-1][:-4]
    def initialize(self):
        self.ax.scatter(self.x, self.y, self.z, s = 40, c = self.colorlist)
        return self.fig,
    def animate(self,i):
        self.ax.view_init(elev=45, azim=3.6*i)
        return self.fig,
    def start(self):
        ani = animation.FuncAnimation(self.fig, self.animate, init_func=self.initialize,
                               frames=90, interval=90, blit=True, repeat=True)
        os.makedirs('rotate_mov', exist_ok=True)
        ani.save(os.path.join('rotate_mov','rotate_'+self.title+'.mp4'), writer="ffmpeg",dpi=100)
        
class Ani3DRotView_v2:
    def __init__(self, data_file, layers, att_file_template):
        feat_data = io_utils.get_one_data(data_file)
        att_data = None
        for layer in layers:
            att_file = att_file_template.replace("layer_name", layer)
            tmp_att_data = io_utils.get_one_data(att_file)
            tmp_att_data = tmp_att_data[:,0]
            att_data = tmp_att_data if att_data is None else att_data + tmp_att_data
        att_data = att_data / len(layers)
        minimum= np.min(att_data)
        maximum= np.max(att_data)
        att_data = att_data[:feat_data.shape[0]]
        colorlist = attention.score_to_color(att_data,minimum,maximum)
        feat_data = feat_data[:len(colorlist)]
        self.fig = plt.figure(figsize = (8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title(data_file.split('/')[-1], size = 20)
        self.ax.set_xlabel("x", size = 14)
        self.ax.set_ylabel("y", size = 14)
        self.ax.set_zlabel("z", size = 14)
        self.x = feat_data[:,0]
        self.y = feat_data[:,1]
        self.z = feat_data[:,2]
        self.colorlist = colorlist
        self.title=data_file.split('/')[-1][:-4]
    def initialize(self):
        self.ax.scatter(self.x, self.y, self.z, s = 40, c = self.colorlist)
        return self.fig,
    def animate(self,i):
        self.ax.view_init(elev=45, azim=3.6*i)
        return self.fig,
    def start(self):
        ani = animation.FuncAnimation(self.fig, self.animate, init_func=self.initialize,
                               frames=90, interval=90, blit=True, repeat=True)
        os.makedirs('rotate_mov', exist_ok=True)
        ani.save(os.path.join('rotate_mov','rotate_'+self.title+'.mp4'), writer="ffmpeg",dpi=100)    
