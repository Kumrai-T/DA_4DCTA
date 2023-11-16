import glob
from tqdm import tqdm
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import sequence
import numpy.random as rnd
from ipywidgets import interact, fixed, Button, HBox, VBox, Dropdown, FloatSlider, Output, Text, Label, Select, IntSlider, BoundedIntText, BoundedFloatText, Textarea, Checkbox
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from PlotLosses import PlotLosses

import project
import feature, io_utils, attention
import LSTM_model as lstm
import viz_utils
import keras

import gc
import time
from datetime import datetime
from keras import backend as K
from tensorflow.keras.utils import Sequence

import tensorflow as tf

class Pipeline:
    def __init__(self, pman):
        self.out_dir = pman.proj_dir
        self.feat_dir = pman.feature_dir()
        self.classes = pman.classes
        self.source_dir = pman.data_dir
        self.pman = pman
        self.out_train = Output(layout={'border': '1px solid red'})
        self.out_result = Output(layout={'border': '1px solid blue'})
        self.vouts = VBox([self.out_train, self.out_result])
        self.targets_dir = pman.targets_dir
        display(self.vouts)
    
    def do_feature_ext_all(self, args):
        dirs = glob.glob(os.path.join(self.pman.data_dir,"*"))
        for one_dir in dirs:
            if os.path.isdir(one_dir):
                ch_one_dir = glob.glob(os.path.join(one_dir,"*"))
                ch_inside_one_dir = glob.glob(os.path.join(ch_one_dir[0],"*"))
                if len(ch_inside_one_dir) != 0:
                    for one_dir2 in ch_one_dir:
                        if os.path.isdir(one_dir2):
                            class_name = os.path.basename(one_dir) + '/' + os.path.basename(one_dir2)
                            if "training" in args:
                                training_list = args["training"]
                            if "testing" in args:
                                testing_list = args["testing"]
                            if os.path.basename(one_dir) in training_list and os.path.basename(one_dir2) in self.classes:
                                class_label = os.path.basename(one_dir2)
                                patient_case = os.path.basename(one_dir)
                                self.do_feature_ext(class_name, class_label, patient_case, args)
                            if os.path.basename(one_dir) in testing_list:
                                class_label = os.path.basename(one_dir2)
                                patient_case = os.path.basename(one_dir)
                                if class_label == 'unclassified':
                                    for i_unclassified in ch_inside_one_dir:
                                        base_unclassified = os.path.basename(i_unclassified)
                                        class_name = os.path.basename(one_dir) + '/' + os.path.basename(one_dir2) + '/' + base_unclassified
                                        class_label = 'test'+patient_case+base_unclassified
                                        self.do_feature_ext(class_name, class_label, patient_case, args)
                                else:
                                    class_label = os.path.basename(one_dir2)+patient_case
                                    self.do_feature_ext(class_name, class_label, patient_case, args)
                else:
                    class_name = os.path.basename(one_dir)
            
    def do_feature_ext(self, class_name, class_label, patient_name, args):
        print("feature extract:", class_name)
        feat_class_dir = os.path.join(self.feat_dir, class_label)
        bw_dir = os.path.normpath(feat_class_dir + os.sep + os.pardir)
        bw_folder_name = os.path.basename(bw_dir)
        if bw_folder_name == 'feature':
            if not os.path.exists(feat_class_dir):
                os.makedirs(feat_class_dir)
        else:
            new_feat_class_dir = os.path.join(self.feat_dir, bw_folder_name)
            if not os.path.exists(new_feat_class_dir):
                os.makedirs(new_feat_class_dir)
            
        for file in tqdm(self.get_file_iterator(class_name)):
            self.feature_extract(file, feat_class_dir, patient_name, args)
            
    def get_file_iterator(self, class_name):
        return glob.glob(os.path.join(self.source_dir, class_name)+'/**/*.csv', recursive=True)
    
    def feature_extract(self, file, feat_class_dir, patient_name, args):
        feature.feature_extraction(file, feat_class_dir, patient_name, args)

    def normal_dist(x, mean, sd):
        prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
        return prob_density
        
    def train(self, pman, nb_epoch, num_node, num_layer, use_dropout, drop_prob, feature_type, eval_method="random", train_dir_base=[], test_case=[], test_dir_base=None, cross_loo_files=None, split_percentage=0.6, balance=False, additional_test_dir_bases=[]): #random, cross_random, cross_leave_one_out, given
        datasetrootdir = self.feat_dir
        targetsrootdir = self.targets_dir
        normal = pman.classes[0]
        mutant = pman.classes[1]
        mutant1 = pman.classes[2]
        mutant2 = pman.classes[3]
        mutant3 = pman.classes[4]
        normal_dir_name = os.path.join(datasetrootdir, normal)
        mutant_dir_name = os.path.join(datasetrootdir, mutant)
        mutant1_dir_name = os.path.join(datasetrootdir, mutant1)
        mutant2_dir_name = os.path.join(datasetrootdir, mutant2)
        mutant3_dir_name = os.path.join(datasetrootdir, mutant3)
        if feature_type == 'speed':
            print("feature: speed")
        elif feature_type == 'acceleration':
            print("feature: acceleration")
        elif feature_type == 'jerk':
            print("feature: jerk")
        elif feature_type == 'speed and acceleration':
            print("feature: speed and acceleration")
        elif feature_type == 'speed and jerk':
            print("feature: speed and jerk")
        elif feature_type == 'acceleration and jerk':
            print("feature: acceleration and jerk")
        else:
            print("feature: speed, acceleration, and jerk")
        print("#epoch:", nb_epoch)
        
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Start time:", current_time)

        start_t = time.perf_counter()

        chk_n = 1
        normal_data_list, mutant_data_list, mutant1_data_list, mutant2_data_list, mutant3_data_list = [], [], [], [], []
        y_normal_data_list, y_mutant_data_list, y_mutant1_data_list, y_mutant2_data_list, y_mutant3_data_list = [], [], [], [], []
        normal_data_before_nor, mutant_data_before_nor, mutant1_data_before_nor, mutant2_data_before_nor, mutant3_data_before_nor = [], [], [], [], []
        
        for one_dir in train_dir_base:
            base_dir = one_dir
            print("base_dir:", base_dir)
            nor_dir_n = normal_dir_name + base_dir
            mu_dir_n = mutant_dir_name + base_dir
            mu1_dir_n = mutant1_dir_name + base_dir
            mu2_dir_n = mutant2_dir_name + base_dir
            mu3_dir_n = mutant3_dir_name + base_dir

            nor_data, F_nor = io_utils.get_data_v4_2(nor_dir_n, feature_type)
            mu_data, F_mu = io_utils.get_data_v4_2(mu_dir_n, feature_type)
            mu1_data, F_mu1 = io_utils.get_data_v4_2(mu1_dir_n, feature_type)
            mu2_data, F_mu2 = io_utils.get_data_v4_2(mu2_dir_n, feature_type)
            mu3_data, F_mu3 = io_utils.get_data_v4_2(mu3_dir_n, feature_type)

            for i in range(nor_data.shape[0]):
                normal_data_before_nor.append(nor_data[i])
            for i in range(mu_data.shape[0]):
                mutant_data_before_nor.append(mu_data[i])
            for i in range(mu1_data.shape[0]):
                mutant1_data_before_nor.append(mu1_data[i])
            for i in range(mu2_data.shape[0]):
                mutant2_data_before_nor.append(mu2_data[i])
            for i in range(mu3_data.shape[0]):
                mutant3_data_before_nor.append(mu3_data[i])

            if chk_n == 1:
                F_normal_list = F_nor
                F_mutant_list = F_mu
                F_mutant1_list = F_mu1
                F_mutant2_list = F_mu2
                F_mutant3_list = F_mu3
            else:
                F_normal_list = np.concatenate((F_normal_list, F_nor))
                F_mutant_list = np.concatenate((F_mutant_list, F_mu))
                F_mutant1_list = np.concatenate((F_mutant1_list, F_mu1))
                F_mutant2_list = np.concatenate((F_mutant2_list, F_mu2))
                F_mutant3_list = np.concatenate((F_mutant3_list, F_mu3))
            chk_n += 1

            nc = np.concatenate(nor_data)
            mc = np.concatenate(mu_data)
            mc1 = np.concatenate(mu1_data)
            mc2 = np.concatenate(mu2_data)
            mc3 = np.concatenate(mu3_data)
            con = np.concatenate((nc, mc, mc1, mc2, mc3))
            
            nor_data, mu_data, mu1_data, mu2_data, mu3_data, mean_n, std_n = io_utils.normalize_list_v2(nor_data, mu_data, mu1_data, mu2_data, mu3_data, bias=0.1)

            y_nor_data = np.full(nor_data.shape[0], 0.0)
            y_mu_data = np.full(mu_data.shape[0], 0.5)
            y_mu1_data = np.full(mu1_data.shape[0], 0.5)
            y_mu2_data = np.full(mu2_data.shape[0], 0.5)
            y_mu3_data = np.full(mu3_data.shape[0], 1.0)

            for i in range(nor_data.shape[0]):
                normal_data_list.append(nor_data[i])
                mutant_data_list.append(mu_data[i])
                mutant1_data_list.append(mu1_data[i])
                mutant2_data_list.append(mu2_data[i])
                mutant3_data_list.append(mu3_data[i])
                
            for i in range(nor_data.shape[0]):
                y_normal_data_list.append(y_nor_data[i])
                y_mutant_data_list.append(y_mu_data[i])
                y_mutant1_data_list.append(y_mu1_data[i])
                y_mutant2_data_list.append(y_mu2_data[i])
                y_mutant3_data_list.append(y_mu3_data[i])

        nor_data = np.array(normal_data_list)
        mu_data = np.array(mutant_data_list)
        mu1_data = np.array(mutant1_data_list)
        mu2_data = np.array(mutant2_data_list)
        mu3_data = np.array(mutant3_data_list)

        y_nor_data = np.array(y_normal_data_list)
        y_mu_data = np.array(y_mutant_data_list)
        y_mu1_data = np.array(y_mutant1_data_list)
        y_mu2_data = np.array(y_mutant2_data_list)
        y_mu3_data = np.array(y_mutant3_data_list)

        F_normal = F_normal_list
        F_mutant = F_mutant_list
        F_mutant1 = F_mutant1_list
        F_mutant2 = F_mutant2_list
        F_mutant3 = F_mutant3_list
        
        if len(additional_test_dir_bases) > 0:
            normal_test = additional_test_dir_bases[0]
            mutant_test = additional_test_dir_bases[1]
            mutant1_test = additional_test_dir_bases[2]
            mutant2_test = additional_test_dir_bases[3]
            mutant3_test = additional_test_dir_bases[4]
            normal_test_dir_name = os.path.join(datasetrootdir, normal_test)
            mutant_test_dir_name = os.path.join(datasetrootdir, mutant_test)
            mutant1_test_dir_name = os.path.join(datasetrootdir, mutant1_test)
            mutant2_test_dir_name = os.path.join(datasetrootdir, mutant2_test)
            mutant3_test_dir_name = os.path.join(datasetrootdir, mutant3_test)
            normal_test_data, _ = io_utils.get_data_v4_2(normal_test_dir_name, feature_type)
            mutant_test_data, _ = io_utils.get_data_v4_2(mutant_test_dir_name, feature_type)
            mutant1_test_data, _ = io_utils.get_data_v4_2(mutant1_test_dir_name, feature_type)
            mutant2_test_data, _ = io_utils.get_data_v4_2(mutant2_test_dir_name, feature_type)
            mutant3_test_data, _ = io_utils.get_data_v4_2(mutant3_test_dir_name, feature_type)
            _, mu_test_data, mu1_test_data, mu2_test_data, _, mean, std = io_utils.normalize_list_v2(normal_test_data, mutant_test_data, mutant1_test_data, mutant2_test_data, mutant3_test_data, bias=0.1)
        else:
            normal_data_before_nor = np.array(normal_data_before_nor)
            mutant_data_before_nor = np.array(mutant_data_before_nor)
            mutant1_data_before_nor = np.array(mutant1_data_before_nor)
            mutant2_data_before_nor = np.array(mutant2_data_before_nor)
            mutant3_data_before_nor = np.array(mutant3_data_before_nor)
            _, _, _, _, _, mean, std = io_utils.normalize_list_v2(normal_data_before_nor, mutant_data_before_nor, mutant1_data_before_nor, mutant2_data_before_nor, mutant3_data_before_nor, bias=0.1)
        normal_data = nor_data
        mutant_data = mu_data
        mutant1_data = mu1_data
        mutant2_data = mu2_data
        mutant3_data = mu3_data

        y_normal_data = y_nor_data
        y_mutant_data = y_mu_data
        y_mutant1_data = y_mu1_data
        y_mutant2_data = y_mu2_data
        y_mutant3_data = y_mu3_data
        del normal_data_before_nor, mutant_data_before_nor, mutant1_data_before_nor, mutant2_data_before_nor, mutant3_data_before_nor, nor_data, mu_data, mu1_data, mu2_data, mu3_data, F_normal_list, F_mutant_list, F_mutant1_list, F_mutant2_list, F_mutant3_list

        print('data loaded')
        print('number of ' + normal + ' files: ' + str(len(normal_data)) + ' ' + str(len(y_normal_data)))
        print('number of ' + mutant3 + ' files: ' + str(len(mutant3_data)) + ' ' + str(len(y_mutant3_data)))
       
        if eval_method == "random":
            pass
        elif eval_method == "cross_random":
            pass
        elif eval_method == "cross_leave_one_out":
            pass
        elif eval_method == "given":
            test_dir_name = os.path.join(datasetrootdir, test_dir_base)
            test_data, F_test = io_utils.get_data_v4_2(test_dir_name, feature_type)
            
            print('number of ' + test_dir_base + ' files: ' + str(len(test_data)))
            print("mean", mean, " std", std)
            if feature_type == 'speed':
                mean = mean[0]
                std = std[0]
            elif feature_type == 'acceleration':
                mean = mean[1]
                std = std[1]
            elif feature_type == 'jerk':
                mean = mean[2]
                std = std[2]
            elif feature_type == 'speed and acceleration':
                mean = mean[0:2]
                std = std[0:2]
            elif feature_type == 'speed and jerk':
                mean = mean[1:3]
                std = std[1:3]
            elif feature_type == 'acceleration and jerk':
                mean = mean[0:3:2]
                std = std[0:3:2]
            else:
                mean = mean
                std = std
            test_data = io_utils.normalize_one(test_data, mean, std, bias=0.1)
            
            window_size = io_utils.get_max_length(normal_data, mutant3_data)
            window_size = max(window_size, io_utils.get_max_length(normal_data, test_data))
            maxlen = window_size
            print('maxlen: ' + str(maxlen))

            n_split_percentage = split_percentage
            m_split_percentage = split_percentage
            if balance:
                if len(normal_data) > len(mutant3_data):
                    n_split_percentage =  len(mutant3_data) * m_split_percentage / len(normal_data)
                else:
                    m_split_percentage =  len(normal_data) * n_split_percentage / len(mutant3_data)

            X_normal_train, X_normal_test, Y_normal_train, Y_normal_test = self.split_data_v2(normal_data, y_normal_data, n_split_percentage)
            X_mutant_train, X_mutant_test, Y_mutant_train, Y_mutant_test = self.split_data_v2(mutant_data, y_mutant_data, m_split_percentage)
            X_mutant1_train, X_mutant1_test, Y_mutant1_train, Y_mutant1_test = self.split_data_v2(mutant1_data, y_mutant1_data, m_split_percentage)
            X_mutant2_train, X_mutant2_test, Y_mutant2_train, Y_mutant2_test = self.split_data_v2(mutant2_data, y_mutant2_data, m_split_percentage)
            X_mutant3_train, X_mutant3_test, Y_mutant3_train, Y_mutant3_test = self.split_data_v2(mutant3_data, y_mutant3_data, m_split_percentage)

            # transform the list to same sequence length
            X_normal_train = sequence.pad_sequences(X_normal_train, maxlen=maxlen, dtype='float16', padding='post',
                                                    truncating='post', value=-1.0)
            X_mutant_train = sequence.pad_sequences(X_mutant_train, maxlen=maxlen, dtype='float16', padding='post',
                                                    truncating='post', value=-1.0)
            X_mutant1_train = sequence.pad_sequences(X_mutant1_train, maxlen=maxlen, dtype='float16', padding='post',
                                                    truncating='post', value=-1.0)
            X_mutant2_train = sequence.pad_sequences(X_mutant2_train, maxlen=maxlen, dtype='float16', padding='post',
                                                    truncating='post', value=-1.0)
            X_mutant3_train = sequence.pad_sequences(X_mutant3_train, maxlen=maxlen, dtype='float16', padding='post',
                                                    truncating='post', value=-1.0)
            X_normal_test = sequence.pad_sequences(X_normal_test, maxlen=maxlen, dtype='float16', padding='post',
                                                   truncating='post', value=-1.0)
            X_mutant_test = sequence.pad_sequences(X_mutant_test, maxlen=maxlen, dtype='float16', padding='post',
                                                   truncating='post', value=-1.0)
            X_mutant1_test = sequence.pad_sequences(X_mutant1_test, maxlen=maxlen, dtype='float16', padding='post',
                                                   truncating='post', value=-1.0)
            X_mutant2_test = sequence.pad_sequences(X_mutant2_test, maxlen=maxlen, dtype='float16', padding='post',
                                                   truncating='post', value=-1.0)
            X_mutant3_test = sequence.pad_sequences(X_mutant3_test, maxlen=maxlen, dtype='float16', padding='post',
                                                   truncating='post', value=-1.0)
            X_test = sequence.pad_sequences(test_data, maxlen=maxlen, dtype='float16', padding='post',
                                                   truncating='post', value=-1.0)

            X_normal_train = np.array(X_normal_train).astype('float16')
            X_mutant_train = np.array(X_mutant_train).astype('float16')
            X_mutant1_train = np.array(X_mutant1_train).astype('float16')
            X_mutant2_train = np.array(X_mutant2_train).astype('float16')
            X_mutant3_train = np.array(X_mutant3_train).astype('float16')
            X_normal_test = np.array(X_normal_test).astype('float16')
            X_mutant_test = np.array(X_mutant_test).astype('float16')
            X_mutant1_test = np.array(X_mutant1_test).astype('float16')
            X_mutant2_test = np.array(X_mutant2_test).astype('float16')
            X_mutant3_test = np.array(X_mutant3_test).astype('float16')
            X_test = np.array(X_test).astype('float16')

            timesteps = window_size
            input_dim = X_normal_train.shape[2]
            batch_size = 16

            skyblue_avg = np.mean(mutant_data, axis=1)
            green_avg = np.mean(mutant1_data, axis=1)
            yellow_avg = np.mean(mutant2_data, axis=1)
            blue_avg = np.mean(normal_data, axis=1)
            red_avg = np.mean(mutant3_data, axis=1)

            skyblue_avg = np.mean(skyblue_avg, axis=0)
            green_avg = np.mean(green_avg, axis=0)
            yellow_avg = np.mean(yellow_avg, axis=0)
            blue_avg = np.mean(blue_avg, axis=0)
            red_avg = np.mean(red_avg, axis=0)

            model = lstm.buildAttentionModelMultiViewCNNLSTMRegression(timesteps, input_dim, blue_avg, skyblue_avg, green_avg, yellow_avg, red_avg, use_dropout=use_dropout, drop_prob=drop_prob, hidden_unit=num_node, num_lstmlayer=num_layer)

            Y_normal_train = np.array(Y_normal_train).astype('float16')
            Y_mutant_train = np.array(Y_mutant_train).astype('float16')
            Y_mutant1_train = np.array(Y_mutant1_train).astype('float16')
            Y_mutant2_train = np.array(Y_mutant2_train).astype('float16')
            Y_mutant3_train = np.array(Y_mutant3_train).astype('float16')
            Y_normal_test = np.array(Y_normal_test).astype('float16')
            Y_mutant_test = np.array(Y_mutant_test).astype('float16')
            Y_mutant1_test = np.array(Y_mutant1_test).astype('float16')
            Y_mutant2_test = np.array(Y_mutant2_test).astype('float16')
            Y_mutant3_test = np.array(Y_mutant3_test).astype('float16')  

            X_train = np.concatenate((X_normal_train, X_mutant3_train))
            Y_train = np.concatenate((Y_normal_train, Y_mutant3_train))
            X_val = np.concatenate((X_normal_test, X_mutant3_test))
            Y_val = np.concatenate((Y_normal_test, Y_mutant3_test))    

            unlabeled_train = np.concatenate((X_mutant_train, X_mutant1_train, X_mutant2_train)) 
            unlabeled_val = np.concatenate((X_mutant_test, X_mutant1_test, X_mutant2_test))

            avg_speed_train = np.mean(X_train[:,:,0], axis=1)
            avg_acc_train = np.mean(X_train[:,:,1], axis=1)

            avg_speed_val = np.mean(X_val[:,:,0], axis=1)
            avg_acc_val = np.mean(X_val[:,:,1], axis=1)

            del X_normal_train, X_mutant_train, X_mutant1_train, X_mutant2_train, X_mutant3_train
            del Y_normal_train, Y_mutant_train, Y_mutant1_train, Y_mutant2_train, Y_mutant3_train
            del X_normal_test, X_mutant_test, X_mutant1_test, X_mutant2_test, X_mutant3_test
            del Y_normal_test, Y_mutant_test, Y_mutant1_test, Y_mutant2_test, Y_mutant3_test
            
            #shuffle
            np.random.seed(0)
            index = np.random.permutation(Y_train.shape[0])
            X_train = X_train[index]
            Y_train = Y_train[index]

            cus_Y_train = np.array([avg_speed_train, avg_acc_train]).transpose()
            cus_Y_val = np.array([avg_speed_val, avg_acc_val]).transpose()

            with self.out_train:
                model_name = "att_model_"+pman.issue_new_model_id()
                plot_losses = PlotLosses()

                bestmodel_name = "att_model_"+pman.issue_new_model_id()
                bestmodel_dir = pman.get_model_dir(bestmodel_name)
                if not os.path.exists(bestmodel_dir):
                    os.makedirs(bestmodel_dir)

                checkpoint = ModelCheckpoint(os.path.join(bestmodel_dir, "best_model_weights.h5"), monitor='val_loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

                model.fit(X_train, #Y_train, 
                         [Y_train, cus_Y_train, cus_Y_train],
                         batch_size=batch_size,
                         epochs=nb_epoch,
                         validation_data=(X_val, [Y_val, cus_Y_val, cus_Y_val]),
#                         validation_data=(X_train, cus_Y_train),
                         callbacks=[plot_losses, checkpoint], verbose=0) #, lr_decay
                model_params = {"timesteps":timesteps, "input_dim":input_dim, "use_dropout":use_dropout, "drop_prob":drop_prob, "num_node":num_node, "num_layer":num_layer}
                model_name = self.save_model(pman, model, eval_method, model_params)

            with self.out_result:

                self.test(model, X_test, None, pman.classes, show_pred_class=False, show_pred_probs=False, save_pred_probs=True, model_name=model_name, test_dir_base=test_dir_base, show_result=False, F_test=F_test)
            
            if len(additional_test_dir_bases) > 0:
                for one_test_dir_base in additional_test_dir_bases:

                    test_dir_name = os.path.join(datasetrootdir, one_test_dir_base)
                    test_data, F_test = io_utils.get_data_v4_2(test_dir_name, feature_type)
                    print('number of ' + one_test_dir_base + ' files: ' + str(len(test_data)))
                    if len(test_data) == 0:
                        continue
                    test_data= io_utils.normalize_one(test_data, mean, std, bias=0.1)
                    X_test = sequence.pad_sequences(test_data, maxlen=maxlen, dtype='float16', padding='post',
                                                   truncating='post', value=-1.0)
                    X_test = np.array(X_test)
                    
                    with self.out_result:
                        self.test(model, X_test, None, pman.classes, show_pred_class=False, show_pred_probs=False, save_pred_probs=True, model_name=model_name, test_dir_base=one_test_dir_base, show_result=False, F_test=F_test)

            model = self.load_best_model(pman, model, model_name)

            self.test(model, X_test, None, pman.classes, show_pred_class=False, show_pred_probs=False, save_pred_probs=True, model_name=model_name, test_dir_base=test_dir_base, show_result=False, F_test=F_test, sub_model='best_model')
            
            if len(additional_test_dir_bases) > 0:
                for one_test_dir_base in additional_test_dir_bases:

                    test_dir_name = os.path.join(datasetrootdir, one_test_dir_base)
                    test_data, F_test = io_utils.get_data_v4_2(test_dir_name, feature_type)
                    print('number of ' + one_test_dir_base + ' files: ' + str(len(test_data)))
                    if len(test_data) == 0:
                        continue
                    test_data= io_utils.normalize_one(test_data, mean, std, bias=0.1)
                    X_test = sequence.pad_sequences(test_data, maxlen=maxlen, dtype='float16', padding='post',
                                                   truncating='post', value=-1.0)
                    X_test = np.array(X_test)
                    
                    with self.out_result:
                        self.test(model, X_test, None, pman.classes, show_pred_class=False, show_pred_probs=False, save_pred_probs=True, model_name=model_name, test_dir_base=one_test_dir_base, show_result=False, F_test=F_test, sub_model='best_model')
        
    def test(self, model, X_test, Y_test, classes, show_pred_class=True, show_pred_probs=False, save_pred_probs=False, model_name=None, test_dir_base=None, show_result=True, F_test=None, sub_model=None):
        predY, _, _ = model.predict(X_test)
        predicted_classes = np.argmax(predY, axis=1)
        Y_test_classes = None
        
        if show_pred_probs:
            print("file,", classes[0], ",", classes[1])
            for fi,probs in zip(F_test, predY):
                print(os.path.basename(fi)+","+str(probs[0]))
        if save_pred_probs:
            result = "file,class \n"
            for fi,probs in zip(F_test, predY):
                result += (os.path.basename(fi)+","+str(probs[0])+"\n")
            
            if sub_model is not None:
                save_dir_best = os.path.join(".", "results",model_name, sub_model, "pred_probs")
                if not os.path.exists(save_dir_best):
                    os.makedirs(save_dir_best)
                save_file = os.path.join(".", "results",model_name, sub_model, "pred_probs", test_dir_base+ ".txt")
            else:
                save_dir = os.path.join(".", "results",model_name, "pred_probs")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_file = os.path.join(".", "results",model_name, "pred_probs", test_dir_base+ ".txt")
            with open(save_file, mode='w') as f:
                f.write(result)

        if show_pred_class:
            print("predicted_classes", predicted_classes)
            Y_test_classes = np.argmax(Y_test, axis=1)
            print("Y_test_classes", Y_test_classes)
        if show_result:
            print(classification_report(Y_test_classes, predicted_classes, 
                    target_names=classes, digits = 6))
            cm = confusion_matrix(Y_test_classes, predicted_classes)
            viz_utils.plot_confusion_matrix(cm, classes=classes, normalize=True)
        return predY, predicted_classes, Y_test_classes
    
    def save_model(self, pman, model, eval_method, val_loss=None, model_params=None, cv_idx=0):
        model_name = "att_model_"+pman.issue_new_model_id()
        model_dir = pman.get_model_dir(model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if model_params is None:
            model_params = {"eval_method":eval_method}
        else:
            model_params.update({"eval_method":eval_method})
        hist = project.TrainingHistory("att_model", model_name, description="eval_method:"+eval_method, dic=model_params)
        self.pman.append_trainhist(hist)
        self.pman.save()
        model.save_weights(os.path.join(model_dir, "model_weights.h5"))
        print("model saved@"+model_dir)
        return model_name
    
    def load_model(self, pman, model_name):
        model_dir = pman.get_model_dir(model_name)
        hist_data = pman.find_train_hist(model_name)
        hist_data.show_info()
        model = lstm.buildAttentionModelMultiViewCNNLSTM(hist_data.dic["timesteps"], hist_data.dic["input_dim"], use_dropout=hist_data.dic["use_dropout"], drop_prob=hist_data.dic["drop_prob"], hidden_unit=hist_data.dic["num_node"], num_lstmlayer=hist_data.dic["num_layer"])
        model.load_weights(os.path.join(model_dir, "model_weights.h5"))
        return model, hist_data

    def load_best_model(self, pman, current_model, model_name):
        model_dir = pman.get_model_dir(model_name)
        current_model.load_weights(os.path.join(model_dir, "best_model_weights.h5"))
        print("best model loaded@"+model_dir+"/best_model_weights.h5")
        return current_model
    
    def save_attention(self, pman, model_name, train_dir_base=[], test=None, model=None, feature_type=None, test_list=[], additional_test_dir_bases=[]):
        if test_list is None:
            test_list = []
        if model is None:
            model, hist_data = self.load_model(pman, model_name)
            timesteps = hist_data.dic["timesteps"]
        else:
            model
        batch_size = 16
        datasetrootdir = self.feat_dir
        normal = pman.classes[0]
        mutant = pman.classes[1]
        mutant1 = pman.classes[2]
        mutant2 = pman.classes[3]
        mutant3 = pman.classes[4]
        normal_dir_name = os.path.join(datasetrootdir, normal)
        mutant_dir_name = os.path.join(datasetrootdir, mutant)
        mutant1_dir_name = os.path.join(datasetrootdir, mutant1)
        mutant2_dir_name = os.path.join(datasetrootdir, mutant2)
        mutant3_dir_name = os.path.join(datasetrootdir, mutant3)
 
        chk_n = 1
        normal_data_list, mutant_data_list, mutant1_data_list, mutant2_data_list, mutant3_data_list = [], [], [], [], []
        y_normal_data_list, y_mutant_data_list, y_mutant1_data_list, y_mutant2_data_list, y_mutant3_data_list = [], [], [], [], []
        normal_data_before_nor, mutant_data_before_nor, mutant1_data_before_nor, mutant2_data_before_nor, mutant3_data_before_nor = [], [], [], [], []

        for one_dir in train_dir_base:
            base_dir = one_dir
            print("base_dir:", base_dir)
            nor_dir_n = normal_dir_name + base_dir
            mu_dir_n = mutant_dir_name + base_dir
            mu1_dir_n = mutant1_dir_name + base_dir
            mu2_dir_n = mutant2_dir_name + base_dir
            mu3_dir_n = mutant3_dir_name + base_dir

            nor_data, F_nor = io_utils.get_data_v4_2(nor_dir_n, feature_type)
            mu_data, F_mu = io_utils.get_data_v4_2(mu_dir_n, feature_type)
            mu1_data, F_mu1 = io_utils.get_data_v4_2(mu1_dir_n, feature_type)
            mu2_data, F_mu2 = io_utils.get_data_v4_2(mu2_dir_n, feature_type)
            mu3_data, F_mu3 = io_utils.get_data_v4_2(mu3_dir_n, feature_type)

            for i in range(nor_data.shape[0]):
                normal_data_before_nor.append(nor_data[i])
            for i in range(mu_data.shape[0]):
                mutant_data_before_nor.append(mu_data[i])
            for i in range(mu1_data.shape[0]):
                mutant1_data_before_nor.append(mu1_data[i])
            for i in range(mu2_data.shape[0]):
                mutant2_data_before_nor.append(mu2_data[i])
            for i in range(mu3_data.shape[0]):
                mutant3_data_before_nor.append(mu3_data[i])

            if chk_n == 1:
                F_normal_list = F_nor
                F_mutant_list = F_mu
                F_mutant1_list = F_mu1
                F_mutant2_list = F_mu2
                F_mutant3_list = F_mu3
            else:
                F_normal_list = np.concatenate((F_normal_list, F_nor))
                F_mutant_list = np.concatenate((F_mutant_list, F_mu))
                F_mutant1_list = np.concatenate((F_mutant1_list, F_mu1))
                F_mutant2_list = np.concatenate((F_mutant2_list, F_mu2))
                F_mutant3_list = np.concatenate((F_mutant3_list, F_mu3))
            chk_n += 1

            nc = np.concatenate(nor_data)
            mc = np.concatenate(mu_data)
            mc1 = np.concatenate(mu1_data)
            mc2 = np.concatenate(mu2_data)
            mc3 = np.concatenate(mu3_data)
            con = np.concatenate((nc, mc, mc1, mc2, mc3))

            nor_data, mu_data, mu1_data, mu2_data, mu3_data, mean_n, std_n = io_utils.normalize_list_v2(nor_data, mu_data, mu1_data, mu2_data, mu3_data, bias=0.1)

            y_nor_data = np.full(nor_data.shape[0], 0.0)
            y_mu_data = np.full(mu_data.shape[0], 0.5)
            y_mu1_data = np.full(mu1_data.shape[0], 0.5)
            y_mu2_data = np.full(mu2_data.shape[0], 0.5)
            y_mu3_data = np.full(mu3_data.shape[0], 1.0)

            for i in range(nor_data.shape[0]):
                normal_data_list.append(nor_data[i])
                mutant_data_list.append(mu_data[i])
                mutant1_data_list.append(mu1_data[i])
                mutant2_data_list.append(mu2_data[i])
                mutant3_data_list.append(mu3_data[i])
                
            for i in range(nor_data.shape[0]):
                y_normal_data_list.append(y_nor_data[i])
                y_mutant_data_list.append(y_mu_data[i])
                y_mutant1_data_list.append(y_mu1_data[i])
                y_mutant2_data_list.append(y_mu2_data[i])
                y_mutant3_data_list.append(y_mu3_data[i])

        nor_data = np.array(normal_data_list)
        mu_data = np.array(mutant_data_list)
        mu1_data = np.array(mutant1_data_list)
        mu2_data = np.array(mutant2_data_list)
        mu3_data = np.array(mutant3_data_list)

        y_nor_data = np.array(y_normal_data_list)
        y_mu_data = np.array(y_mutant_data_list)
        y_mu1_data = np.array(y_mutant1_data_list)
        y_mu2_data = np.array(y_mutant2_data_list)
        y_mu3_data = np.array(y_mutant3_data_list)

        F_normal = F_normal_list
        F_mutant = F_mutant_list
        F_mutant1 = F_mutant1_list
        F_mutant2 = F_mutant2_list
        F_mutant3 = F_mutant3_list
        
        if len(additional_test_dir_bases) > 0:
            normal_test = additional_test_dir_bases[0]
            mutant_test = additional_test_dir_bases[1]
            mutant1_test = additional_test_dir_bases[2]
            mutant2_test = additional_test_dir_bases[3]
            mutant3_test = additional_test_dir_bases[4]
            normal_test_dir_name = os.path.join(datasetrootdir, normal_test)
            mutant_test_dir_name = os.path.join(datasetrootdir, mutant_test)
            mutant1_test_dir_name = os.path.join(datasetrootdir, mutant1_test)
            mutant2_test_dir_name = os.path.join(datasetrootdir, mutant2_test)
            mutant3_test_dir_name = os.path.join(datasetrootdir, mutant3_test)
            normal_test_data, _ = io_utils.get_data_v4_2(normal_test_dir_name, feature_type)
            mutant_test_data, _ = io_utils.get_data_v4_2(mutant_test_dir_name, feature_type)
            mutant1_test_data, _ = io_utils.get_data_v4_2(mutant1_test_dir_name, feature_type)
            mutant2_test_data, _ = io_utils.get_data_v4_2(mutant2_test_dir_name, feature_type)
            mutant3_test_data, _ = io_utils.get_data_v4_2(mutant3_test_dir_name, feature_type)
            _, mu_test_data, mu1_test_data, mu2_test_data, _, mean, std = io_utils.normalize_list_v2(normal_test_data, mutant_test_data, mutant1_test_data, mutant2_test_data, mutant3_test_data, bias=0.1)
        else:
            normal_data_before_nor = np.array(normal_data_before_nor)
            mutant_data_before_nor = np.array(mutant_data_before_nor)
            mutant1_data_before_nor = np.array(mutant1_data_before_nor)
            mutant2_data_before_nor = np.array(mutant2_data_before_nor)
            mutant3_data_before_nor = np.array(mutant3_data_before_nor)
            _, _, _, _, _, mean, std = io_utils.normalize_list_v2(normal_data_before_nor, mutant_data_before_nor, mutant1_data_before_nor, mutant2_data_before_nor, mutant3_data_before_nor, bias=0.1)
        normal_data = nor_data
        mutant_data = mu_data
        mutant1_data = mu1_data
        mutant2_data = mu2_data
        mutant3_data = mu3_data

        y_normal_data = y_nor_data
        y_mutant_data = y_mu_data
        y_mutant1_data = y_mu1_data
        y_mutant2_data = y_mu2_data
        y_mutant3_data = y_mu3_data
        del normal_data_before_nor, mutant_data_before_nor, mutant1_data_before_nor, mutant2_data_before_nor, mutant3_data_before_nor, nor_data, mu_data, mu1_data, mu2_data, mu3_data, F_normal_list, F_mutant_list, F_mutant1_list, F_mutant2_list, F_mutant3_list
        
        window_size = io_utils.get_max_length(normal_data, mutant_data)
        if test is not None:
            test_dir_name = os.path.join(datasetrootdir, test)
            test_data, F_test = io_utils.get_data(test_dir_name)
            window_size = max(window_size, io_utils.get_max_length(normal_data, test_data))
            
        maxlen = window_size
        print('maxlen: ' + str(maxlen))
        timesteps = window_size

        split_percentage = 0.6
        balance = False
        n_split_percentage = split_percentage
        m_split_percentage = split_percentage
        if balance:
            if len(normal_data) > len(mutant3_data):
                n_split_percentage =  len(mutant3_data) * m_split_percentage / len(normal_data)
            else:
                m_split_percentage =  len(normal_data) * n_split_percentage / len(mutant3_data)

        # transform the list to same sequence length
        X_normal_train = sequence.pad_sequences(normal_data, maxlen=maxlen, dtype='float16', padding='post',
                                                truncating='post', value=-1.0)
        X_mutant_train = sequence.pad_sequences(mutant_data, maxlen=maxlen, dtype='float16', padding='post',
                                                truncating='post', value=-1.0)
        X_mutant1_train = sequence.pad_sequences(mutant1_data, maxlen=maxlen, dtype='float16', padding='post',
                                                truncating='post', value=-1.0)
        X_mutant2_train = sequence.pad_sequences(mutant2_data, maxlen=maxlen, dtype='float16', padding='post',
                                                truncating='post', value=-1.0)
        X_mutant3_train = sequence.pad_sequences(mutant3_data, maxlen=maxlen, dtype='float16', padding='post',
                                                truncating='post', value=-1.0)

        X_normal_train = np.array(X_normal_train)
        X_mutant_train = np.array(X_mutant_train)
        X_mutant1_train = np.array(X_mutant1_train)
        X_mutant2_train = np.array(X_mutant2_train)
        X_mutant3_train = np.array(X_mutant3_train)

        normal_save_dir = pman.get_attention_dir(model_name, normal)
        mutant_save_dir = pman.get_attention_dir(model_name, mutant)
        mutant1_save_dir = pman.get_attention_dir(model_name, mutant1)
        mutant2_save_dir = pman.get_attention_dir(model_name, mutant2)
        mutant3_save_dir = pman.get_attention_dir(model_name, mutant3)        
        
        if test is not None:
            test_list.append(test)

        if feature_type == 'speed':
            mean = mean[0]
            std = std[0]
        elif feature_type == 'acceleration':
            mean = mean[1]
            std = std[1]
        elif feature_type == 'jerk':
            mean = mean[2]
            std = std[2]
        elif feature_type == 'speed and acceleration':
            mean = mean[0:2]
            std = std[0:2]
        elif feature_type == 'speed and jerk':
            mean = mean[1:3]
            std = std[1:3]
        elif feature_type == 'acceleration and jerk':
            mean = mean[0:3:2]
            std = std[0:3:2]
        else:
            mean = mean
            std = std
            
        for test in test_list:
            test_dir_name = os.path.join(datasetrootdir, test)
            test_data, F_test = io_utils.get_data_v4_2(test_dir_name, feature_type)
            
            print('number of ' + test + ' files: ' + str(len(test_data)))
            print("mean", mean, " std", std)
            test_data= io_utils.normalize_one(test_data, mean, std, bias=0.1)

            X_test = sequence.pad_sequences(test_data, maxlen=maxlen, dtype='float16', padding='post',
                                                   truncating='post', value=-1.0)
            X_test = np.array(X_test)
            
            test_save_dir = pman.get_attention_dir(model_name, test)
            attention.write_intermediate_output(model, X_test, batch_size, F_test, test_save_dir, timesteps,  savebinary=True)
        
    
    def split_data(self, data, split_percentage): #override if necessary
        return io_utils.splitData_by_random(data, split_percentage)

    def split_data_v2(self, data, targets, split_percentage): #override if necessary
        return io_utils.splitData_by_random_v2(data, targets, split_percentage)
        
