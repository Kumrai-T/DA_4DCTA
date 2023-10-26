import os
import numpy as np
from keras import backend as K
from tensorflow.python.keras.backend import eager_learning_phase_scope

def get_intermediate_output(model, X, batch_size, layer_index, timesteps):
    # get the output of an intermediate layer

    layer_output_dic = {}
    for ly in layer_index:
        if model.layers[ly].output_shape[1] is None:
            layer_output_dic[model.layers[ly].name] = np.empty((0, timesteps,) + model.layers[ly].output_shape[2:])
        else:
            layer_output_dic[model.layers[ly].name] = np.empty((0,) + model.layers[ly].output_shape[1:])

    get_output = K.function([model.layers[0].input],
                            [model.layers[ly].output for ly in layer_index])
    with eager_learning_phase_scope(value=0):
        for i in range(0, len(X), batch_size):
            layer_outputs = get_output([X[i: i + batch_size]])
            for j in range(len(layer_outputs)):
                layer_output_dic[model.layers[layer_index[j]].name] = np.concatenate(
                    (layer_output_dic[model.layers[layer_index[j]].name], layer_outputs[j]))

    return layer_output_dic


def write_intermediate_output(model, normal_data, batch_size, normal_files,
                              normal_save_dir, timesteps, layers=['lstm', 'attention', 'conv1d'],#, 'dense', 'merge'],#['lstm', 'timedistributed'],
                              savebinary=False):
    layerlist = []
    
    for ly in range(len(model.layers)):
        for layer in layers:
            if layer in model.layers[ly].name:
                layerlist.append(ly)
                print('write ' + model.layers[ly].name)

    # for normal
    layer_output_dic = get_intermediate_output(model, normal_data, batch_size, layerlist, timesteps)

    for k in list(layer_output_dic.keys()):
        dirname = os.path.join(normal_save_dir, k)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        if savebinary:
            for i in range(len(layer_output_dic[k])):
                np.save(os.path.join(dirname, os.path.split(normal_files[i])[1].replace('.csv', '')),
                        layer_output_dic[k][i])
        else:
            for i in range(len(layer_output_dic[k])):
                np.savetxt(os.path.join(dirname, os.path.split(normal_files[i])[1]), layer_output_dic[k][i],
                           delimiter=',')
                
        print('output of ' + k + ' layer was written.')
        
        
def score_to_color(scorelist, minimum=None, maximum=None):
    
    if minimum is None:
        minimum=np.min(scorelist)
    if maximum is None:
        maximum=np.max(scorelist)
    if maximum > minimum:
        sl = (scorelist-minimum) / (maximum-minimum)
    else:
        sl = scorelist-minimum
        
    sorted_sl = sorted(sl)
    
    colorlist=[0]*len(sl)
    colorlist_sorted=[0]*len(sorted_sl)
    for i in range(len(sl)):
        if sl[i]<0.0:
            colorlist[i]=[0.0, 0.0, 1.0]
        elif sl[i]<0.25:
            colorlist[i]=[0.0, sl[i]*4, 1.0]
        elif sl[i]<0.5:
            colorlist[i]=[0.0, 1.0, 2.0-sl[i]*4]
        elif sl[i]<0.75:
            colorlist[i]=[sl[i]*4-2.0, 1.0, 0.0]
        elif sl[i]<1.0:
            colorlist[i]=[1.0, 4.0-sl[i]*4, 0.0]
        else:
            colorlist[i]=[1.0, 0.0, 0.0]

    for i in range(len(sorted_sl)):
        if sorted_sl[i]<0.0:
            colorlist_sorted[i]=[0.0, 0.0, 1.0]
        elif sorted_sl[i]<0.25:
            colorlist_sorted[i]=[0.0, sorted_sl[i]*4, 1.0]
        elif sorted_sl[i]<0.5:
            colorlist_sorted[i]=[0.0, 1.0, 2.0-sorted_sl[i]*4]
        elif sorted_sl[i]<0.75:
            colorlist_sorted[i]=[sorted_sl[i]*4-2.0, 1.0, 0.0]
        elif sorted_sl[i]<1.0:
            colorlist_sorted[i]=[1.0, 4.0-sorted_sl[i]*4, 0.0]
        else:
            colorlist_sorted[i]=[1.0, 0.0, 0.0]
    
    return colorlist, colorlist_sorted
