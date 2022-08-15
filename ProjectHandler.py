from copy import deepcopy
from glob import glob
from shutil import copyfile
from PyQt5.QtWidgets import QMessageBox
from pandas.io import pickle
from Utility import show_dialoge
import pickle
from PyQt5.QtCore import QObject, pyqtSignal
import os
import Config
import Class

class ProjectHandler(QObject):

    open_project_signal = pyqtSignal(dict, bool);
    set_project_name_signal = pyqtSignal(str);
    new_project_setup_signal = pyqtSignal();

    def __init__(self):
        super().__init__();
        self.__current_project_name = "";
        self.__projects_root = "";

        #create file to store most recent project
        if not os.path.exists('mrp.uog'):
            open('mrp.uog','w');
        pass
    
    @property
    def current_project_name(self):
        return self.__current_project_name;
    
    @current_project_name.setter
    def current_project_name(self, name):
        self.__current_project_name = name;

    ''''
        Use given path to create a folder for the new project
        we assume that the name given is the name of project
    '''
    def new_project(self, path):
        proj_name = os.path.basename(path);
        #Create a folder at the given path
        os.makedirs(path);
        #Create all necessary folders for the new project in all folders
        os.makedirs(os.path.sep.join([path,'images']));
        os.makedirs(os.path.sep.join([path,'labels']));
        os.makedirs(os.path.sep.join([path, 'ckpts']));
        os.makedirs(os.path.sep.join([path, 'evaluation']));
        os.makedirs(os.path.sep.join([path, 'experiments']));

        #Set most recent project to the currently created one
        self.__current_project_name = proj_name;
        Config.PROJECT_NAME = self.__current_project_name;
        Config.PROJECT_ROOT = path;
        self.__projects_root = Config.PROJECT_ROOT;
        self.new_project_setup_signal.emit();
        self.save_project(False);
        self.save_project_meta(list());
        self.set_project_name_signal.emit(proj_name);

        pass

    def save_project(self, show_dialoge_bool = True):
        pickle.dump(Class.data_pool_handler.data_list,open(os.path.sep.join([self.__projects_root, self.__current_project_name + '.uog']),'wb'));
        #df = pd.DataFrame(list(d.items()), columns = ['image_path','state', 'type']);
        #df.to_pickle();
        if show_dialoge_bool:
            show_dialoge(QMessageBox.Icon.Information, f"Successfully saved","Info", QMessageBox.Ok);
        self.save_most_recent_project();
        pass
    
    def save_project_as(self, path):
        proj_name = os.path.basename(path);
        base_path = path[:path.rfind('/')];
        #Create a folder at the given path
        os.makedirs(path);
        #Create all necessary folders for the new project in all folders
        os.makedirs(os.path.sep.join([path,'images']));
        os.makedirs(os.path.sep.join([path,'labels']));
        os.makedirs(os.path.sep.join([path, 'ckpts']));
        os.makedirs(os.path.sep.join([path, 'evaluation']));
        os.makedirs(os.path.sep.join([path, 'experiments']));

        #Copy project file name
        copyfile(os.path.sep.join([self.__projects_root, self.__current_project_name + '.uog']),
        os.path.sep.join([path,proj_name+'.uog']));

        copyfile(os.path.sep.join([self.__projects_root, self.__current_project_name + '.meta']),
        os.path.sep.join([path,proj_name+'.meta']));

        #Copy each folder data to the destination folder
        lstdir = os.listdir(os.path.sep.join([self.__projects_root,'images']));
        for i in lstdir:
            copyfile(os.path.sep.join([self.__projects_root,'images',i]), 
                os.path.sep.join([path,'images',i]))
        
        lstdir = os.listdir(os.path.sep.join([self.__projects_root,'labels']));
        for i in lstdir:
            copyfile(os.path.sep.join([self.__projects_root,'labels',i]), 
                os.path.sep.join([path,'labels',i]))
        
        lstdir = os.listdir(os.path.sep.join([self.__projects_root,'ckpts']));
        for i in lstdir:
            copyfile(os.path.sep.join([self.__projects_root,'ckpts',i]), 
                os.path.sep.join([path,'ckpts',i]))
        
        lstdir = os.listdir(os.path.sep.join([self.__projects_root,'evaluation']));
        for i in lstdir:
            copyfile(os.path.sep.join([self.__projects_root,'evaluation',i]), 
                os.path.sep.join([path,'evaluation',i]))
        
        lstdir = os.listdir(os.path.sep.join([self.__projects_root,'experiments']));
        for i in lstdir:
            copyfile(os.path.sep.join([self.__projects_root,'experiments',i]), 
                os.path.sep.join([path,'experiments',i]))

        self.new_project_setup_signal.emit();

        self.__current_project_name = proj_name;
        Config.PROJECT_NAME = self.__current_project_name;
        Config.PROJECT_ROOT = path;
        self.__projects_root = Config.PROJECT_ROOT;

        self.set_project_name_signal.emit(proj_name);
        data_list = pickle.load(open(os.path.sep.join([path,proj_name + '.uog']),'rb'));\
        self.open_project_signal.emit(data_list, False);

        show_dialoge(QMessageBox.Icon.Information, f"Project successfully saved to new path","Save succeed", QMessageBox.Ok);

        pass

    def save_project_meta(self, meta):
        pickle.dump(meta, open(os.path.sep.join([self.__projects_root, self.__current_project_name + '.meta']),'wb'));

    def get_project_meta(self):
        meta = pickle.load(open(os.path.sep.join([self.__projects_root,   
        self.__current_project_name + '.meta']),'rb'));
        return meta;
    
    def update_project_meta_slot(self, name):
        m = self.get_project_meta();
        if len(m) == 0:
            m.append([name]);
        else:
            m[0].append(name);
        
        self.save_project_meta(m);
    
    def get_project_names_slot(self):
        m = self.get_project_meta();
        if len(m) != 0:
            for l in m[0]:
                Config.PROJECT_PREDEFINED_NAMES[0].append(l);
        pass

    def save_most_recent_project(self):
        mrp = open('mrp.uog','w');
        mrp.write(os.path.sep.join([self.__projects_root, self.__current_project_name + '.uog']));
        mrp.close();
        pass
    
    def open_project(self, path = None):

        self.new_project_setup_signal.emit();
        #If no name given, open most recent project
        if path == None:
            mrp = open('mrp.uog','r');
            project_path = mrp.read();
            if os.path.exists(project_path):
                #First clear every loaded item from previous project
                self.new_project_setup_signal.emit();
                proj_name = os.path.basename(project_path);
                proj_name = proj_name[:proj_name.rfind('.')];
                self.__current_project_name = proj_name;
                self.__projects_root = project_path[:project_path.rfind('\\')];
                Config.PROJECT_NAME = self.__current_project_name;
                Config.PROJECT_ROOT = self.__projects_root;

                data_list = pickle.load(open(project_path,'rb'));
                # data_list['71.jpeg'] = data_list['71_1.jpeg'];
                #data_list.pop('199.png');
                # meta = pickle.load(open('C:\\PhD\\Miscellaneous\\Spine and Ribs\\labels\\303.meta', 'rb'));
                # meta['Ribs'] = meta['Vertebra'];
                # meta.pop('Vertebra');
                # pickle.dump(meta, open('C:\\PhD\\Miscellaneous\\Spine and Ribs\\labels\\303.meta', 'wb'))
                # data_list['303.jpeg']


                tmp_datalist, change = self.__check_for_unload_images(data_list);

                if change is True:
                    msgBox = QMessageBox()
                    msgBox.setIcon(QMessageBox.Icon.Warning)
                    msgBox.setText("Your project is not in sync with your local image database. Do you want to update your directory now?")
                    msgBox.setWindowTitle("Confirmation")
                    msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No);

                    return_value = msgBox.exec()

                    if return_value == QMessageBox.Yes:
                        data_list = deepcopy(tmp_datalist);

                self.save_most_recent_project();

                self.open_project_signal.emit(data_list, True);
                self.set_project_name_signal.emit(proj_name);

                return True;
            else:
                #If most recent project not found, start a new project.
                return False;
        else:
            if os.path.exists(path):
                #First clear every loaded item from previous project
                self.new_project_setup_signal.emit();

                data_list = pickle.load(open(path,'rb'));
                    
                #We assume that the filename is the project name
                project_name = os.path.basename(path);
                project_name = project_name[:project_name.rfind('.')];

                self.__current_project_name = project_name;
                Config.PROJECT_NAME = self.__current_project_name;

                project_root = path[:path.rfind('/')];
                Config.PROJECT_ROOT = project_root;
                self.__projects_root = project_root;

                #set this project as the most recent project
                self.save_most_recent_project();

                self.set_project_name_signal.emit(project_name);
                self.open_project_signal.emit(data_list, True);

                
                return True;
            else:
                show_dialoge(QMessageBox.Icon.Critical, f"Project couldn't be found","Error", QMessageBox.Ok);
        pass

    def __check_for_unload_images(self, data_list):
        '''
            Here we check labels folder to check for images that
            have not been added to datalist of the project and 
            ask user about them
        '''

        temp_data_list = deepcopy(data_list);

        img_lst = glob(os.path.sep.join([Config.PROJECT_ROOT, "images"]) + "\\*");

        change = False;
        for l in img_lst:
            file_name = os.path.basename(l);
            path_to_meta = os.path.sep.join([Config.PROJECT_ROOT, 'labels', file_name[:file_name.rfind(".")] + ".meta"]);
            if (file_name not in data_list and os.path.exists(path_to_meta)):
                temp_data_list[file_name] = ['labeled', 'image'];
                change = True;
            elif (file_name in data_list and data_list[file_name][0] == 'unlabeled' and os.path.exists(path_to_meta)):
                temp_data_list[file_name][0] = 'labeled';
                change = True;

        return temp_data_list, change;
    
    def __relod_dataset(self):
        '''
            Here we match every image with labels that are availabe.
            We also match each label in meta file and correct any naming issues.
        '''

        temp_data_list = dict();

        img_lst = glob(os.path.sep.join([Config.PROJECT_ROOT, "images"]) + "\\*");

        for l in img_lst:
            file_name = os.path.basename(l);
            
            file_name_we = file_name[:file_name.rfind(".")];

            path_to_meta = os.path.sep.join([Config.PROJECT_ROOT, 'labels', file_name_we + ".meta"]);
            if (os.path.exists(path_to_meta) is False):
                print(f"Could not find meta in reload dataset: {path_to_meta}");
            else:
                temp_data_list[file_name] = ['labeled', 'image'];
                meta_data = pickle.load(open(path_to_meta, 'rb'));
                for k in meta_data.keys():
                    if k != 'rot' and k!= 'exp':
                        d = meta_data[k];
                        name = d[2];
                        tmp_name = name[len(name)-6:];
                        new_name = file_name_we+tmp_name;
                        meta_data[k][2] = new_name;
                pickle.dump(meta_data, open(path_to_meta, 'wb'));
        return temp_data_list;

