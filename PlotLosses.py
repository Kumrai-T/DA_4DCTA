import keras
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation
from keras.callbacks import LearningRateScheduler

import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output

#added 20220727 Kumrai
import gc
from keras import backend as K

# updatable plot
# a minimal example (sort of)

class PlotLosses(keras.callbacks.Callback):
    '''
    Lossのグラフプロット用
    '''
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.title("Learning progress")
        plt.savefig('losses.png')
        plt.show();

        gc.collect()
        
def PlotLossAndAccInHistoryStatic(history):
    print("training history")
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    for one_key in history.history.keys():
        if one_key.endswith("loss"):
            vals = history.history[one_key]
            epochs = range(len(vals))
            ax1.plot(epochs, vals, label=one_key)
    ax1.set_title('Loss history')
    plt.legend()
    plt.show();
