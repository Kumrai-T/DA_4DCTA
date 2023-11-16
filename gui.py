import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import io_utils, attention, viz_utils
import re
from IPython.display import clear_output

from ipywidgets import interact, fixed, Button, HBox, VBox, Dropdown, FloatSlider, Output, Text, Label, Select, IntSlider, BoundedIntText, BoundedFloatText, Textarea, Checkbox

class FeatureAttViz:
    def __init__(self, pman, model_name, target_patient=None, test_dir_name=None):
        self.pman = pman
        self.model_name = model_name
        self.class_val = None
        self.layer_val = None
        self.file_name = None
        self.att_layers = []
        
        layers = []
        if target_patient is None:
            classes = pman.classes if test_dir_name is None else pman.classes + [test_dir_name]
            for lay in os.listdir(pman.get_attention_dir(model_name, classes[0])):
                if lay.startswith("attentionl_") or lay.startswith("attentionc_"):
                    layers.append(lay)
            if len(layers)==0:#one-layer network
                for lay in os.listdir(pman.get_attention_dir(model_name, classes[0])):
                    if lay.startswith("attentionllast_") or lay.startswith("attentionclast_") or lay.startswith("dense_"):
                        layers.append(lay)
                    if lay.startswith("attentionllast_") or lay.startswith("attentionclast_"):
                        self.att_layers.append(lay)
        else:
            classes = []
            for cl in pman.classes:
                classes.append(cl+target_patient)
            print(classes)
            for lay in os.listdir(pman.get_attention_dir(model_name, classes[0])):
                if lay.startswith("attentionl_") or lay.startswith("attentionc_"):
                    layers.append(lay)
            if len(layers)==0:#one-layer network
                for lay in os.listdir(pman.get_attention_dir(model_name, classes[0])):
                    if lay.startswith("attentionllast_") or lay.startswith("attentionclast_") or lay.startswith("dense_"):
                        layers.append(lay)
                    if lay.startswith("attentionllast_") or lay.startswith("attentionclast_"):
                        self.att_layers.append(lay)

        self.class_sel = Select(options=classes, description="Class")
        self.class_sel.observe(self.on_value_change_class_sel, names='value')
        self.layer_sel = Select(options=layers, description="Layer")
        self.layer_sel.observe(self.on_value_change_layer_sel, names='value')
        self.data_sel = Select(options=[], description="Data")
        self.data_sel.observe(self.on_value_change_data_sel, names="value")
        vbox = VBox([self.class_sel, self.layer_sel, self.data_sel])
        display(vbox)
        self.on_value_change_class_sel({"new":self.class_sel.options[0]})
        self.on_value_change_layer_sel({"new":self.layer_sel.options[0]})

        self.button_one = Button(description="One")
        self.button_one.on_click(self.button_one_on)
        self.button_all = Button(description="All")
        self.button_all.on_click(self.button_all_on)
        self.button_sum_attention = Button(description="Sum_Att")
        self.button_sum_attention.on_click(self.button_sum_attention_on)
        self.button_avg_attention = Button(description="Avg_Att")
        self.button_avg_attention.on_click(self.button_avg_attention_on)
        hbox = HBox([self.button_one, self.button_all, self.button_sum_attention, self.button_avg_attention])
        display(hbox)

    
    def button_one_on(self, b):
        if self.class_val is not None and self.layer_val is not None and self.file_name is not None:
            print(self.class_val, self.layer_val, self.file_name)
            att_file = os.path.join(self.pman.get_attention_dir(self.model_name, self.class_val), self.layer_val, self.file_name)
#            data_file = os.path.join( self.pman.feature_dir(), self.class_val, self.file_name)
            #edit 20210914 Kumrai
            data_file = os.path.join( self.pman.feature_dir(), self.class_val, os.path.split(self.file_name)[1].replace('.npy', '.csv'))
            print("att_file", att_file)
            print("data_file", data_file)
            viz_utils.plot_att_feat_graph(data_file, att_file)
    
    def button_all_on(self, b):
        if self.class_val is not None and self.file_name is not None:
            print(self.class_val, self.file_name)
#            data_file = os.path.join( self.pman.feature_dir(), self.class_val, self.file_name)
            data_file = os.path.join( self.pman.feature_dir(), self.class_val, os.path.split(self.file_name)[1].replace('.npy', '.csv'))
            att_file = os.path.join( self.pman.get_attention_dir(self.model_name, self.class_val), "layer_name", self.file_name)
            print("data_file", data_file)
            print("att_file", att_file)
            viz_utils.plot_all_att_feat_graph(data_file, self.layer_sel.options, att_file)

    def button_sum_attention_on(self, b):
        if self.class_val is not None and self.file_name is not None:
            print(self.class_val, self.file_name)
            data_file = os.path.join( self.pman.feature_dir(), self.class_val, os.path.split(self.file_name)[1].replace('.npy', '.csv'))
            att_layers_list = []
            if len(self.att_layers) > 0:
                for att_l in self.att_layers:
                    att_file = os.path.join(self.pman.get_attention_dir(self.model_name, self.class_val), att_l, self.file_name)
                    att_layers_list.append(att_file)
            viz_utils.plot_sum_att_feat_graph(data_file, att_layers_list)

    def button_avg_attention_on(self, b):
        if self.class_val is not None and self.file_name is not None:
            print(self.class_val, self.file_name)
            data_file = os.path.join( self.pman.feature_dir(), self.class_val, os.path.split(self.file_name)[1].replace('.npy', '.csv'))
            att_layers_list = []
            if len(self.att_layers) > 0:
                for att_l in self.att_layers:
                    att_file = os.path.join(self.pman.get_attention_dir(self.model_name, self.class_val), att_l, self.file_name)
                    att_layers_list.append(att_file)
            viz_utils.plot_avg_att_feat_graph(data_file, att_layers_list)        

    def on_value_change_class_sel(self, change):
        self.class_val = change['new']
        if self.class_val is not None and self.layer_val is not None:
            self.update_data_sel()
        
    def on_value_change_layer_sel(self, change):
        self.layer_val = change['new']
        if self.class_val is not None and self.layer_val is not None:
            self.update_data_sel()
        
    def update_data_sel(self):
        att_dir = os.path.join( self.pman.get_attention_dir(self.model_name, self.class_val), self.layer_val)
        #self.data_sel.unobserve(self.on_value_change_data_sel, names="value")
        self.data_sel.options = os.listdir(att_dir)
        #self.data_sel.observe(self.on_value_change_data_sel, names="value")
        
    def on_value_change_data_sel(self, change):
        self.file_name = change['new']


class AttV3Diz:
    def __init__(self, pman, model_name, test_dir_name=None):
        self.pman = pman
        self.model_name = model_name
        self.class_val = None
        self.layer_val = None
        self.file_name = None
        classes = pman.classes if test_dir_name is None else pman.classes + [test_dir_name]
        layers = []
        for lay in os.listdir(pman.get_attention_dir(model_name, classes[0])):
            if lay.startswith("attentionl_") or lay.startswith("attentionc_"):
                layers.append(lay)
        if len(layers)==0:#one-layer network
            for lay in os.listdir(pman.get_attention_dir(model_name, classes[0])):
                if lay.startswith("attentionllast_") or lay.startswith("attentionclast_"):
                    layers.append(lay)  
        self.class_sel = Select(options=classes, description="Class")
        self.class_sel.observe(self.on_value_change_class_sel, names='value')
        self.layer_sel = Select(options=layers, description="Layer")
        self.layer_sel.observe(self.on_value_change_layer_sel, names='value')
        self.data_sel = Select(options=[], description="Data")
        self.data_sel.observe(self.on_value_change_data_sel, names="value")
        vbox = VBox([self.class_sel, self.layer_sel, self.data_sel])
        display(vbox)
        self.on_value_change_class_sel({"new":self.class_sel.options[0]})
        self.on_value_change_layer_sel({"new":self.layer_sel.options[0]})
        
        self.button_one = Button(description="One attention")
        self.button_one.on_click(self.button_one_on)
        self.button_all = Button(description="All attention")
        self.button_all.on_click(self.button_all_on)
        
        hbox = HBox([self.button_one, self.button_all])
        display(hbox)
    
    def button_one_on(self, b):
        #return
        if self.class_val is not None and self.layer_val is not None and self.file_name is not None:
            print(self.class_val, self.layer_val, self.file_name)
            att_file = os.path.join(self.pman.get_attention_dir(self.model_name, self.class_val), self.layer_val, self.file_name)
            data_file = os.path.join(self.pman.data_dir, self.class_val, self.file_name)
            print("att_file", att_file)
            print("data_file", data_file)
            #viz_utils.plot_att_feat_graph(data_file, att_file)
    
    def button_all_on(self, b):
        if self.class_val is not None and self.file_name is not None:
            print(self.class_val, self.file_name)
            data_file = os.path.join( self.pman.data_dir, self.class_val, self.file_name)
            if not os.path.exists(data_file):
#                data_files = glob.glob(os.path.join( self.pman.data_dir, self.class_val, '**', self.file_name), recursive=True)
                first_dir = self.file_name
                print("first_dir", first_dir)
                str_name = []
                if 'case' in first_dir:
                    check_f = 0
                    for i in self.file_name:
                        if i == "_":
                            check_f += 1
                        if check_f < 2:
                            str_name.append(i)
                        else:
                            break
                else:
                    check_f = 0
                    for i in self.file_name:
                        if i == "_":
                            check_f += 1
                        if check_f < 1:
                            str_name.append(i)
                        else:
                            break
                first_dir = ''.join([str(elem) for elem in str_name])
                last_filename = os.path.split(self.file_name)[1].replace('.npy', '.csv')
                print("dirs:", first_dir, last_filename)
                if 'test' not in self.class_val and len(self.class_val) < 5:
                    data_files = glob.glob(os.path.join( self.pman.data_dir, self.class_val, '**', first_dir, last_filename), recursive=True)
                else:
                    data_files = glob.glob(os.path.join( self.pman.data_dir, self.class_val, '**', last_filename), recursive=True)
                print("data_files:", data_files)
                if len(data_files) == 0:
                    print('error data file does not exist:', data_file)
                    return
                elif len(data_files) > 1:
                    print('warning multiple data files found:', data_files)
                    data_file = data_files[0]
                else:
                    data_file = data_files[0]
            att_file = os.path.join( self.pman.get_attention_dir(self.model_name, self.class_val), "layer_name", self.file_name)

            print("data_file", data_file)
            print("att_file", att_file)
            viz_utils.plot_all_att_3D(data_file, self.layer_sel.options, att_file)
            ani3 = viz_utils.Ani3DRotView(data_file, self.layer_sel.options, att_file)
            ani3.start()
        
    def on_value_change_class_sel(self, change):
        self.class_val = change['new']
        if self.class_val is not None and self.layer_val is not None:
            self.update_data_sel()
        
    def on_value_change_layer_sel(self, change):
        self.layer_val = change['new']
        if self.class_val is not None and self.layer_val is not None:
            self.update_data_sel()
        
    def update_data_sel(self):
        att_dir = os.path.join( self.pman.get_attention_dir(self.model_name, self.class_val), self.layer_val)
        #self.data_sel.unobserve(self.on_value_change_data_sel, names="value")
        self.data_sel.options = os.listdir(att_dir)
        #self.data_sel.observe(self.on_value_change_data_sel, names="value")
        
    def on_value_change_data_sel(self, change):
        self.file_name = change['new']

class AttV3Diz_v2:
    def __init__(self, pman, model_name, target_patient=None, test_dir_name=None):
        self.pman = pman
        self.model_name = model_name
        self.class_val = None
        self.layer_val = None
        self.file_name = None
        self.att_layers = []
        self.target_patient = target_patient

        layers = []
        if target_patient is None:
            classes = pman.classes if test_dir_name is None else pman.classes + [test_dir_name]
            for lay in os.listdir(pman.get_attention_dir(model_name, classes[0])):
                if lay.startswith("attentionl_") or lay.startswith("attentionc_"):
                    layers.append(lay)
            if len(layers)==0:#one-layer network
                for lay in os.listdir(pman.get_attention_dir(model_name, classes[0])):
                    if lay.startswith("attentionllast_") or lay.startswith("attentionclast_") or lay.startswith("dense_"):
                        layers.append(lay)
                    if lay.startswith("attentionllast_") or lay.startswith("attentionclast_"):
                        self.att_layers.append(lay)
        else:
            classes = []
            for cl in pman.classes:
                classes.append(cl+target_patient)
            print(classes)
            for lay in os.listdir(pman.get_attention_dir(model_name, classes[0])):
                if lay.startswith("attentionl_") or lay.startswith("attentionc_"):
                    layers.append(lay)
            if len(layers)==0:#one-layer network
                for lay in os.listdir(pman.get_attention_dir(model_name, classes[0])):
                    if lay.startswith("attentionllast_") or lay.startswith("attentionclast_") or lay.startswith("dense_"):
                        layers.append(lay)
                    if lay.startswith("attentionllast_") or lay.startswith("attentionclast_"):
                        self.att_layers.append(lay)

        self.class_sel = Select(options=classes, description="Class")
        self.class_sel.observe(self.on_value_change_class_sel, names='value')
        self.layer_sel = Select(options=layers, description="Layer")
        self.layer_sel.observe(self.on_value_change_layer_sel, names='value')
        self.data_sel = Select(options=[], description="Data")
        self.data_sel.observe(self.on_value_change_data_sel, names="value")
        vbox = VBox([self.class_sel, self.layer_sel, self.data_sel])
        display(vbox)
        self.on_value_change_class_sel({"new":self.class_sel.options[0]})
        self.on_value_change_layer_sel({"new":self.layer_sel.options[0]})
        
        self.button_one = Button(description="One attention")
        self.button_one.on_click(self.button_one_on)
        self.button_all = Button(description="All attention")
        self.button_all.on_click(self.button_all_on)
        self.button_avg_all = Button(description="Avg all attention")
        self.button_avg_all.on_click(self.button_avg_all_on)
        self.button_avg_all_traj = Button(description="Avg all attention_all_traj")
        self.button_avg_all_traj.on_click(self.button_avg_all_traj_on)
        self.button_sum_all = Button(description="Sum all attention")
        self.button_sum_all.on_click(self.button_sum_all_on)
        
        hbox = HBox([self.button_one, self.button_all, self.button_avg_all, self.button_avg_all_traj, self.button_sum_all])
        display(hbox)
    
    def button_one_on(self, b):
        #return
        if self.class_val is not None and self.layer_val is not None and self.file_name is not None:
            print(self.class_val, self.layer_val, self.file_name)
            att_file = os.path.join(self.pman.get_attention_dir(self.model_name, self.class_val), self.layer_val, self.file_name)
            data_file = os.path.join(self.pman.data_dir, self.class_val, self.file_name)
            print("att_file", att_file)
            print("data_file", data_file)
            #viz_utils.plot_att_feat_graph(data_file, att_file)
    
    def button_all_on(self, b):
        if self.class_val is not None and self.file_name is not None:
            print(self.class_val, self.file_name, self.target_patient[0][0])
            target_class = ""            
            for cha in self.class_val:
                if cha != self.target_patient[0][0]:
                    target_class += cha
                else:
                    break
            print("test:", target_class)
            data_file = os.path.join( self.pman.data_dir, self.target_patient, target_class, self.file_name)
            print("data_file_button_all_on:", data_file)
            if not os.path.exists(data_file):
                #print("data_files:", data_files)
                if len(data_file) == 0:
                    print('error data file does not exist:', data_file)
                    return
                #elif len(data_file) > 1:
                #    print('warning multiple data files found:', data_file)
                #    data_file = data_file
                else:
                    data_file = data_file
            #att_file = os.path.join( self.pman.get_attention_dir(self.model_name, self.class_val), "layer_name", self.file_name)
            att_layers_list = []
            if len(self.att_layers) > 0:
                for att_l in self.att_layers:
                    att_file = os.path.join(self.pman.get_attention_dir(self.model_name, self.class_val), att_l, self.file_name)
                    att_layers_list.append(att_file)

            print("data_file", data_file)
            print("att_file", att_layers_list)
            viz_utils.plot_all_att_3D(data_file, self.layer_sel.options, att_layers_list)
            ani3 = viz_utils.Ani3DRotView_v2(data_file, self.layer_sel.options, att_layers_list)
            ani3.start()

    def button_avg_all_on(self, b):
        if self.class_val is not None and self.file_name is not None:
            #clear_output(wait=True)
            print(self.class_val, self.file_name, self.target_patient[0][0])
            target_class = ""            
            for cha in self.class_val:
                if cha != self.target_patient[0][0]:
                    target_class += cha
                else:
                    break
            print("test:", target_class)
            data_file = os.path.join( self.pman.data_dir, self.target_patient, target_class, self.file_name)
            print("data_file_button_all_on:", data_file, data_file.replace('.npy', '.csv'))
            if not os.path.exists(data_file):
                #print("data_files:", data_files)
                if len(data_file) == 0:
                    print('error data file does not exist:', data_file)
                    return
                #elif len(data_file) > 1:
                #    print('warning multiple data files found:', data_file)
                #    data_file = data_file
                else:
                    data_file = data_file.replace('.npy', '.csv')
            #att_file = os.path.join( self.pman.get_attention_dir(self.model_name, self.class_val), "layer_name", self.file_name)
            att_layers_list = []
            if len(self.att_layers) > 0:
                for att_l in self.att_layers:
                    att_file = os.path.join(self.pman.get_attention_dir(self.model_name, self.class_val), att_l, self.file_name)
                    att_layers_list.append(att_file)

            #print("data_file", data_file)
            #print("att_file", att_layers_list)
            viz_utils.plot_avg_all_att_3D(data_file, self.layer_sel.options, att_layers_list)
            #ani3 = viz_utils.Ani3DRotView_v2(data_file, self.layer_sel.options, att_layers_list)
            #ani3.start()

    def button_avg_all_traj_on(self, b):
        att_dir = os.path.join(self.pman.get_attention_dir(self.model_name, self.class_val), self.layer_val)
        data_file_list = sorted(os.listdir(att_dir), key=lambda s: int(re.search(r'\d+', s).group()))
        print(len(data_file_list))
        for f_name in data_file_list:
            if self.class_val is not None and self.file_name is not None:
                #clear_output(wait=True)
                print(self.class_val, f_name, self.target_patient[0][0])
                target_class = ""            
                for cha in self.class_val:
                    if cha != self.target_patient[0][0]:
                        target_class += cha
                    else:
                        break
                #print("test:", target_class)
                data_file = os.path.join( self.pman.data_dir, self.target_patient, target_class, f_name)
                #print("data_file_button_all_on:", data_file, data_file.replace('.npy', '.csv'))
                if not os.path.exists(data_file):
                    #print("data_files:", data_files)
                    if len(data_file) == 0:
                        print('error data file does not exist:', data_file)
                        return
                    #elif len(data_file) > 1:
                    #    print('warning multiple data files found:', data_file)
                    #    data_file = data_file
                    else:
                        data_file = data_file.replace('.npy', '.csv')
                #att_file = os.path.join( self.pman.get_attention_dir(self.model_name, self.class_val), "layer_name", self.file_name)
                att_layers_list = []
                if len(self.att_layers) > 0:
                    for att_l in self.att_layers:
                        att_file = os.path.join(self.pman.get_attention_dir(self.model_name, self.class_val), att_l, f_name)
                        att_layers_list.append(att_file)

                print("data_file", data_file)
                #print("att_file", att_layers_list)
                viz_utils.plot_avg_all_att_3D_all_traj(data_file, self.layer_sel.options, att_layers_list)


    def button_sum_all_on(self, b):
        if self.class_val is not None and self.file_name is not None:
            print(self.class_val, self.file_name)
            target_class = ""            
            for cha in self.class_val:
                if cha != self.target_patient[0][0]:
                    target_class += cha
                else:
                    break
            print("test:", target_class)
            data_file = os.path.join( self.pman.data_dir, self.target_patient, target_class, self.file_name)
            print("data_file_button_all_on:", data_file, data_file.replace('.npy', '.csv'))
            if not os.path.exists(data_file):
                #print("data_files:", data_files)
                if len(data_file) == 0:
                    print('error data file does not exist:', data_file)
                    return
                #elif len(data_file) > 1:
                #    print('warning multiple data files found:', data_file)
                #    data_file = data_file
                else:
                    data_file = data_file.replace('.npy', '.csv')
            #att_file = os.path.join( self.pman.get_attention_dir(self.model_name, self.class_val), "layer_name", self.file_name)
            att_layers_list = []
            if len(self.att_layers) > 0:
                for att_l in self.att_layers:
                    att_file = os.path.join(self.pman.get_attention_dir(self.model_name, self.class_val), att_l, self.file_name)
                    att_layers_list.append(att_file)

            print("data_file", data_file)
            viz_utils.plot_sum_all_att_3D(data_file, self.layer_sel.options, att_layers_list)
        
    def on_value_change_class_sel(self, change):
        self.class_val = change['new']
        if self.class_val is not None and self.layer_val is not None:
            self.update_data_sel()
        
    def on_value_change_layer_sel(self, change):
        self.layer_val = change['new']
        if self.class_val is not None and self.layer_val is not None:
            self.update_data_sel()
        
    def update_data_sel(self):
        att_dir = os.path.join(self.pman.get_attention_dir(self.model_name, self.class_val), self.layer_val)
        #self.data_sel.unobserve(self.on_value_change_data_sel, names="value")
        self.data_sel.options = sorted(os.listdir(att_dir), key=lambda s: int(re.search(r'\d+', s).group()))
        #self.data_sel.observe(self.on_value_change_data_sel, names="value")
        
    def on_value_change_data_sel(self, change):
        self.file_name = change['new']
