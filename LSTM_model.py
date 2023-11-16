import keras
from keras.layers import Input, CuDNNLSTM, Dense, Dropout, Activation, noise, normalization, TimeDistributed, Flatten, Masking, Embedding, Conv1D, RepeatVector, Permute, Lambda, merge, concatenate, multiply, LSTM
from keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.losses import mse, mae
import os
from RevIN import RevIN
import torch
import tensorflow.keras.backend as K

import numpy as np
import tensorflow as tf
import constant_value as const

from tensorflow.python.framework.ops import enable_eager_execution
enable_eager_execution()

_model_name = 'lstm_model'


def model_path(modelrootdir, tag='', prefix='', sufix=''):
    return os.path.join(modelrootdir, tag, prefix + _model_name + sufix + '.hdf5')

def make_deepconv_layers(_input, n_layers, hidden_unit, timesteps, use_dropout=True, kernel_enlarge=True, kernel_init_len=0.05, layer_suffix=""):
    min_kernel_size = 4
    kernel_size = max(int(timesteps*kernel_init_len), min_kernel_size)
    #print("kernel_size:",kernel_size)
    sent_representations = []
    convs =[]
    for i in range(n_layers):
        _kernel_size =(kernel_size*(i+1)) if kernel_enlarge else (kernel_size)
        conv_l = Conv1D(hidden_unit, _kernel_size, padding='same',  activation='tanh', name="conv1d"+layer_suffix+"_"+str(i))(_input if len(convs)==0 else convs[-1])
        if use_dropout:
            conv_l = Dropout(0.5)(conv_l)
        convs.append(conv_l)
        # compute importance for each step
        attention = TimeDistributed(Dense(1, activation='tanh'))(conv_l) 
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(hidden_unit)(attention)
        attention = Permute([2, 1], name="attentionc"+("last" if i==n_layers-1 else "")+layer_suffix+"_"+str(i))(attention)
        # apply the attention
        sent_representation = multiply([conv_l, attention])
#        sent_representation = merge([conv_l, attention], mode='mul')
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
        sent_representations.append(sent_representation)
    return convs, sent_representations

def make_deepconv_layers_without_attention(_input, n_layers, hidden_unit, timesteps, use_dropout=True, kernel_enlarge=True, kernel_init_len=0.05, layer_suffix=""):
    min_kernel_size = 4
    kernel_size = max(int(timesteps*kernel_init_len), min_kernel_size)
    #print("kernel_size:",kernel_size)
    sent_representations = []
    convs =[]
    for i in range(n_layers):
        _kernel_size =(kernel_size*(i+1)) if kernel_enlarge else (kernel_size)
        conv_l = Conv1D(hidden_unit, _kernel_size, padding='same',  activation='tanh', name="conv1d"+layer_suffix+"_"+str(i))(_input if len(convs)==0 else convs[-1])
        if use_dropout:
            conv_l = Dropout(0.5)(conv_l)
        conv_l = Lambda(lambda xin: K.sum(xin, axis=1))(conv_l)
        convs.append(conv_l)
    return convs

def make_lstm_layers(_input, n_layers, hidden_unit, use_dropout, layer_suffix=""):
    sent_representations = []
    lstms = []
    for i in range(n_layers):
#        lstm_l = LSTM(hidden_unit, return_sequences=True, name="lstm"+layer_suffix+"_"+str(i))(_input if len(lstms)==0 else lstms[-1])
        lstm_l = CuDNNLSTM(hidden_unit, return_sequences=True, name="lstm"+layer_suffix+"_"+str(i))(_input if len(lstms)==0 else lstms[-1])
        if use_dropout:
            lstm_l = Dropout(0.5)(lstm_l)
#            lstm_ll = CuDNNLSTM(hidden_unit, return_sequences=False)(lstm_l)
#            lstm_ll = LSTM(hidden_unit, return_sequences=False)(lstm_l)
        lstms.append(lstm_l)
        # compute importance for each step
        attention = TimeDistributed(Dense(1, activation='tanh'))(lstm_l) 
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(hidden_unit)(attention)
        attention = Permute([2, 1], name="attentionl"+("last" if i==n_layers-1 else "")+layer_suffix+"_"+str(i))(attention)
        # apply the attention
        sent_representation = multiply([lstm_l, attention])
#        sent_representation = merge([lstm_l, attention], mode='mul')
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
        sent_representations.append(sent_representation)
    return lstms, sent_representations

def make_lstm_layers_without_attention(_input, n_layers, hidden_unit, use_dropout, layer_suffix=""):
    sent_representations = []
    lstms = []
    for i in range(n_layers):
#        lstm_l = LSTM(hidden_unit, return_sequences=True, name="lstm"+layer_suffix+"_"+str(i))(_input if len(lstms)==0 else lstms[-1])
        lstm_l = CuDNNLSTM(hidden_unit, return_sequences=False, name="lstm"+layer_suffix+"_"+str(i))(_input if len(lstms)==0 else lstms[-1])
        if use_dropout:
#            lstm_l = CuDNNLSTM(hidden_unit, return_sequences=False)(lstm_l)
            lstm_l = Dropout(0.5)(lstm_l)            
#            lstm_ll = LSTM(hidden_unit, return_sequences=False)(lstm_l)
        lstms.append(lstm_l)
    return lstms
    
def make_lstm_layer(_input, n_layers, hidden_unit, use_dropout, layer_suffix=""):
    #activations = Bidirectional(LSTM(hidden_unit, return_sequences=True), name="lstm_"+str(n_layers-1))(_input)
    activations = LSTM(hidden_unit, return_sequences=True, name="lstm"+layer_suffix+"_"+str(n_layers-1))(_input)
    if use_dropout:
        activations = Dropout(0.5)(activations)

    # compute importance for each step
    attention = TimeDistributed(Dense(1, activation='tanh'))(activations) 
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(hidden_unit)(attention)
    #attention = RepeatVector(hidden_unit*2)(attention)
    attention = Permute([2, 1], name="attention"+layer_suffix+"_"+str(n_layers-1))(attention)

    # apply the attention
    sent_representation = multiply([activations, attention])
#    sent_representation = merge([activations, attention], mode='mul')
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
    return sent_representation

def buildAttentionModelMultiViewCNNLSTMRegression(timesteps, input_dim, blue_avg, skyblue_avg, green_avg, yellow_avg, red_avg,  # input setting
                   hidden_unit=8, num_lstmlayer=2, use_batch_normalize=False, use_dropout=True, drop_prob=0.5, 
                   # layer setting
                   lr=0.001, decay=0.001  # optimizer setting
                   ):

    def dist_diff(a, b):
        return K.sqrt(K.sum(K.square(a - b), axis=-1))

    def dist_patient_diff(y_true, y_pred):
        dists = [dist_diff(y_pred, skyblue_avg), dist_diff(y_pred, green_avg), dist_diff(y_pred, yellow_avg), dist_diff(y_pred, blue_avg), dist_diff(y_pred, red_avg)]
        mins = tf.reduce_min(dists, axis=0)
        dist_loss = tf.reduce_mean(mins)
        l2_norm = tf.norm(y_true-y_pred, ord='euclidean')
        return l2_norm + tf.cast(dist_loss, tf.float32)

    def dist_unlabeled(y_true, y_pred):
        dists_class = [dist_diff(y_pred, red_avg), dist_diff(y_pred, blue_avg)]
        mins_class = tf.reduce_min(dists_class, axis=0)
        dis_class_loss = tf.reduce_mean(mins_class)
        l2_norm = tf.norm(y_true-y_pred, ord='euclidean')
        return l2_norm + tf.cast(dis_class_loss, tf.float32)

    scales = 8
    kernel_exp_step = 1
    kernel_init_len=0.03
    print('scale:'+str(scales) + ' num_lstmlayer:'+str(num_lstmlayer)+' hidden_unit' + str(hidden_unit))
    all_sent_representations = []
    _input = Input(shape=(timesteps, input_dim))
    masking = TimeDistributed(Masking(mask_value=-1.0))(_input)
    for sidx in range(scales):
        if sidx < int(scales/2):
            m_kernel_init_len = kernel_init_len*(int(sidx/kernel_exp_step)+1)
            m_num_layers = num_lstmlayer#+(scales-1-sidx)
            convs, sent_representations = make_deepconv_layers(masking, m_num_layers, hidden_unit, timesteps, use_dropout, kernel_enlarge=False, kernel_init_len=m_kernel_init_len, layer_suffix="_"+str(sidx))
            #all_sent_representations.append(sent_representations[-1])
            all_sent_representations = all_sent_representations + sent_representations
        else:
            m_num_layers = num_lstmlayer#+(scales-1-sidx)
            lstms, lstm_representations = make_lstm_layers(_input ,m_num_layers ,hidden_unit, use_dropout,"_"+str(sidx))
            #all_sent_representations.append(lstm_representations[-1])
            all_sent_representations = all_sent_representations + lstm_representations
            
    if len(all_sent_representations) > 1:
#        merge_sent_representations = merge(all_sent_representations, mode='concat')
        merge_sent_representations = concatenate(all_sent_representations)
    else:
        merge_sent_representations = all_sent_representations[0]
#    dropout_l = Dropout(0.15)(merge_sent_representations)
#    _output = Dense(1, activation='linear')(dropout_l)
    patient_ind = Dense(2, activation='linear', name='patient_indie')(merge_sent_representations)
    class_ind = Dense(2, activation='linear', name='class_indie')(merge_sent_representations)
    concat1 = concatenate([patient_ind, class_ind])
    _output = Dense(1, activation='linear', name='pred')(concat1)
    model = Model(inputs=_input, outputs=[_output, patient_ind, class_ind])
    opt = Adam(learning_rate=lr)

    model.compile(optimizer=opt, loss={'pred':mae, 'patient_indie':dist_patient_diff, 'class_indie':dist_unlabeled}, run_eagerly=True)
    plot_model(model, to_file='AttentionModelMultiViewCNNLSTMRegression.png', show_shapes=True, show_layer_names=True)
    return model

def buildMultiViewLSTMRegression(timesteps, input_dim, blue_avg, skyblue_avg, green_avg, yellow_avg, red_avg,  # input setting
                   hidden_unit=8, num_lstmlayer=2, use_batch_normalize=False, use_dropout=True, drop_prob=0.5, 
                   # layer setting
                   lr=0.001, decay=0.001  # optimizer setting
                   ):

    scales = 2
    kernel_exp_step = 1
    kernel_init_len=0.03
    print('scale:'+str(scales) + ' num_lstmlayer:'+str(num_lstmlayer)+' hidden_unit' + str(hidden_unit))
    all_sent_representations = []
    _input = Input(shape=(timesteps, input_dim))
    masking = TimeDistributed(Masking(mask_value=-1.0))(_input)
    for sidx in range(scales):
        m_num_layers = num_lstmlayer#+(scales-1-sidx)
        lstms = make_lstm_layers_without_attention(_input ,m_num_layers ,hidden_unit, use_dropout,"_"+str(sidx))
            #all_sent_representations.append(lstm_representations[-1])
        all_sent_representations = all_sent_representations + lstms #+lstm_representations
            
    if len(all_sent_representations) > 1:
#        merge_sent_representations = merge(all_sent_representations, mode='concat')
        merge_sent_representations = concatenate(all_sent_representations)
    else:
        merge_sent_representations = all_sent_representations[0]
    _output = Dense(1, activation='linear')(merge_sent_representations)
    model = Model(inputs=_input, outputs=_output)
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='mean_absolute_error', metrics=['mean_absolute_error'], run_eagerly=True)
    plot_model(model, to_file='MultiViewLSTMRegression.png', show_shapes=True, show_layer_names=True)
    return model

def set_weight(model, weights):
    model.set_weights(weights)
    return model


def print_modelconfig(model):
    configstr = ''
    for layer_conf in model.get_config():
        configstr += layer_conf['class_name'] + '(' + layer_conf['config']['name'] + ')'
        configstr += '\n'
        for key in list(layer_conf['config'].keys()):
            if key != 'name':
                if type(layer_conf['config'][key]) == dict:
                    configstr += '\t' + layer_conf['config'][key]['class_name'] + '(' + \
                                 layer_conf['config'][key]['config']['name'] + ')'
                    configstr += '\n'
                    for key2 in list(layer_conf['config'][key]['config'].keys()):
                        if key2 != 'name':
                            configstr += '\t\t' + key2 + ': ' + str(layer_conf['config'][key]['config'][key2])
                            configstr += '\n'
                else:
                    configstr += '\t' + key + ': ' + str(layer_conf['config'][key])
                    configstr += '\n'

        configstr += '\n'

    print(configstr)

    return configstr


if __name__ == '__main__':
    from . import io_utils

    datasetrootdir, resultrootdir, modelrootdir, normal, mutant, savebinary, train_params = io_utils.arg_parse(
        'print model')
    tag = normal + '_vs_' + mutant
    modelp = model_path(modelrootdir, tag=tag)
    if os.path.exists(modelp):
        print('loading model...')
    model = load_model(modelp)
    print_modelconfig(model)
