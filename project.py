import glob
import pickle
import os, sys

class ProjectManager:
    file_name = "project.pickle"
    def __init__(self, data_dir, proj_dir, targets_dir, classes=None):
        self.data_dir = data_dir
        self.proj_dir = proj_dir
        self.targets_dir = targets_dir
        self.data_histories = []
        self.train_histories = []
        dirs = glob.glob(os.path.join(self.data_dir,"*"))
        self.classes = []
        if len(dirs) != 2:
            print("Warning: ", len(dirs), "dirs are found in ", self.data_dir)
            if len(classes)==2:
                self.classes = classes
                print("provided classes are set:", self.classes)
            else:
                self.classes = classes
                print("[multiple classes] provided classes are set:", self.classes)
        else:
            for one_dir in dirs:
                self.classes.append(os.path.basename(one_dir))
            print("classes:", self.classes)

    def feature_dir(self):
        return os.path.join(self.proj_dir, "feature")       

    def result_dir(self, modelname):
        return os.path.join(self.proj_dir, "result", modelname)    

    def targets_dir(self):
        return os.path.join(self.proj_dir, "targets")  
   
    def save(self):
        if not os.path.exists(self.proj_dir):
            os.makedirs(self.proj_dir)
        save_file = os.path.join(self.proj_dir, ProjectManager.file_name)
        with open(save_file, mode='wb') as f:
            pickle.dump(self, f)
            
        return save_file
    
    def append_datahist(self, hist):
        self.data_histories.append(hist)
    
    def append_trainhist(self, hist):
        self.train_histories.append(hist)
        
    def get_source_dir(self, dirname):
        return os.path.join(self.proj_dir, dirname)

    def get_model_dir(self, modelname):
        return os.path.join(self.proj_dir, "modeldir", modelname)

    def get_attention_dir(self, modelname, class_name):
        return os.path.join(self.proj_dir, "attention", modelname, class_name)

    def find_train_hist(self, model_name):
        for hist in self.train_histories:
            if hist.key == "att_model" and hist.value == model_name:
                return hist
        print("warning hist not found:", model_name)
        for hist in self.train_histories:
            hist.show_info()
        return None
    
    def list_source_dict(self):
        results = {}
        for dhist in self.data_histories:
            results[dhist.value + ": " + str(dhist.description)] = dhist.value
        return results
    
    def enumerate_models(self):
        models = []
        for dhist in self.train_histories:
            if dhist.key == "att_model":
                models.append(dhist.value)
        return models
        
    def issue_new_model_id(self):
        ids = []
        for dhist in self.train_histories:
            if dhist.key == "att_model":
                sp = dhist.value.split("_")
                ids.append(int(sp[-1]))
        if len(ids) > 0:
            ids.sort()
            return str(ids[-1] + 1)
        else:
            return "0"
    
    def issue_new_data_dir_id(self):
        ids = []
        for dhist in self.data_histories:
            sp = dhist.value.split("_")
            ids.append(int(sp[-1]))
        if len(ids) > 0:
            ids.sort()
            return str(ids[-1] + 1)
        else:
            return "0"
        
    def show_info(self):
        print("data dir:", self.data_dir)
        print("project dir:", self.proj_dir)
        print("classes:",self.classes)
    
    @staticmethod
    def load(proj_dir):
        file_name = os.path.join(proj_dir, ProjectManager.file_name)
        if not os.path.exists(file_name):
            return None
        with open(file_name, mode='rb') as f:
            return pickle.load(f)

class DataHistory:
    def __init__(self, key, value, description=None, dic=None):
        self.key = key
        self.value = value
        self.description = description
        self.dic = dic
    def show_info(self):
        print(self.key, self.value, self.description, self.dic)
    
class TrainingHistory:
    def __init__(self, key, value, description=None, dic=None):
        self.key = key
        self.value = value
        self.description = description
        self.dic = dic
    def show_info(self):
        print(self.key, self.value, self.description, self.dic)
