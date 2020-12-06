from functools import wraps
from time import time
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn import  metrics, model_selection
from sklearn.model_selection import train_test_split
from datetime import datetime
from csci_utils.io.io import atomic_write



# A decorator to check on the timings of the each major methods used
def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        te = time()
        print ('Class method :%r took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap


class Printresults:
    """The class is used for the printing the figure and csv file for the resulted classification"""
    def __init__(self,model,model_name,y_test,y_pred,df_pca,names,image_shape,result_path,test_flag):
        self.model=model
        self.model_name=model_name
        self.y_test=y_test
        self.y_pred=y_pred
        self.df_pca=df_pca
        self.result_path=result_path
        self.test_flag  = test_flag
        self.names=names  # names of the classes being used
        self.image_shape=image_shape # X,Y dimension of the image being processed

    @timing
    def print_reports(self):
        """Method to print all the reports """
        data = confusion_matrix(self.y_test, self.y_pred)
        df_cm = pd.DataFrame(data, columns=np.unique(self.names), index = np.unique(self.names))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        plt.figure(figsize = (10,8))
        title1= 'Confusion matrix for ' + self.model_name
        sns.set(font_scale=1.4)#for label size
        # suffix file with the timestamp
        timestr=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        suffix='_confusion_map_'+timestr+'.png'
        fig_name=f'{self.result_path}{self.model_name}{suffix}'
        sns.heatmap(df_cm, cmap="viridis", annot=True,annot_kws={"size": 16}, fmt='d').set_title(title1)
        # save the classification heatmap
        plt.savefig(fig_name, dpi=300)
        print(classification_report(self.y_test, self.y_pred, target_names = self.names))
        report = classification_report(self.y_test, self.y_pred, target_names = self.names, output_dict=True)
        # frame the dataframe object for the classification report
        df = pd.DataFrame(report).transpose()
        report_file=self.model_name+'_clf_report'+timestr+'.csv'
        filename=f'{self.result_path}{report_file}'

        # get the atomic writer setup
        if os.path.exists(filename) == False :
        # atomically write csv file by getting the path where to write it
            with atomic_write(filename, "w", False) as f:
                dir_path = f
            if self.test_flag=="No":
                new_file=f'{dir_path}/results/{report_file}'
            else:
                new_file=f'{dir_path}/test/results/{report_file}'
            df.to_csv(new_file)
        self.plot_clasf_map()


    def plot_clasf_map(self):
        """Method for finding the classification map of the classifier"""
        lis=[]
        for i in range(self.df_pca.shape[0]):
            if self.df_pca.iloc[i, -1] == 0:
              lis.append(0)
            else:
              lis.append(self.model.predict(self.df_pca.iloc[i, :-1].values.reshape(1, -1)))
        clmap = np.array(lis).reshape(self.image_shape[0], self.image_shape[1]).astype('float')
        plt.figure(figsize=(10, 8))
        plt.imshow(clmap, cmap='nipy_spectral')
        plt.colorbar()
        plt.title( self.model_name + ' classification map')
        plt.axis('off')
        # Time stamp each result file
        timestr=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        suffix= '_IP_cmap_'+timestr+'.png'
        fig_name=f'{self.result_path}{self.model_name}{suffix}'
        plt.savefig(fig_name)


class PerformPCA:
    """ Class which will perform Principal component analysis"""
    def __init__(self,df,num_components):
        self.df=df  # dataframe whose dimension to be reduced
        self.num_components = num_components # how many principal components are needed

    # Program to Create list  with integers within given range
    # I need to do this because of some strange error caused by range() function
    def createfileList(self,r1,r2):
        """ Fuction to create a list of the numbers """
        r1=int(r1)
        r2=int(r2)
        if r1 == r2:
            return r1
        else:
            # Create empty list to store file numbers
            res = []
            while r1 < r2+1 :
                res.append(r1)
                r1 += 1
            return res

    def perform_pca(self):
        """Function to perform PCA for the given number of components"""
        pca = PCA(n_components = self.num_components)  # create PCA object
        dtemp = pca.fit_transform(self.df.iloc[:, :-1].values) # get predictors PCs
        y=self.df.iloc[:, -1].values # get ground truth
        df_pca = pd.concat([pd.DataFrame(data = dtemp), pd.DataFrame(data = y)], axis = 1)
        list2=self.createfileList(1,self.num_components)
        df_pca.columns = [f'PC-{i}' for i in list2]+['classes']
        return df_pca



class SplitafterPCA(PerformPCA):
    """ The class will split train test data after he PCA is performed"""

    def __init__(self,df,num_components, test_size,random_state):
        super().__init__(df,num_components)
        self.test_size=test_size  #train test split size
        self.random_state=random_state


    def split_train_test(self):
        """ Function to perform split on the training and test data"""
        df_pca=super().perform_pca()
        # remove the zero level class from the dataset as it does not indicate anything
        df_temp = df_pca[df_pca['classes']!=0]
        X_new = df_temp.iloc[:,:-1].values
        y_new = df_temp.iloc [:,-1].values
        X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=self.test_size, random_state=self.random_state)
        return df_pca, X_train, X_test, y_train, y_test


class Classifers:
    """ The main classifer class creates models for classification"""

    def __init__(self,clf_name, dict_param):
        self.clf_name=clf_name # variable to hold classifier name
        self.dict_param=dict_param # variable to hold param of classification

    def create_model(self):
        """ Method to create classification model of choice"""
        if self.clf_name== "RF":
            ntree=self.dict_param['n_estimators']
            max_depth=self.dict_param['max_depth']
            max_features=self.dict_param['max_features']
            model_name = "Random_Forest_Classifier"
            model = RandomForestClassifier(n_estimators=ntree, max_depth=max_depth, max_features=max_features)
        elif self.clf_name == "SVC":
            C=self.dict_param['C'] # numeric value for regularizing param
            kernel=self.dict_param['kernel'] # string value for kernel
            cache_size=self.dict_param['cache_size']
            #example: SVC(C = 100000, kernel = 'rbf', cache_size = 10*1024)
            model= SVC(C = C, kernel = kernel, cache_size = cache_size)
            model_name = "Support_Vector_Classifier"
            return model,model_name
        else:
            print("Unsupported classifier model type")
            raise AttributeError()

        return model,model_name


class Fit_predict:
    """ The class which does actual fitting and prediction """
    def __init__(self,model, X_train,y_train,X_test,y_test):
        self.model=model
        self.X_train=X_train
        self.X_test=X_test
        self.y_test=y_test
        self.y_train=y_train


    def multiclass_roc_auc_score(self,y_pred):
        """Method that will auc_score for the multiclass classification"""
        lb = preprocessing.LabelBinarizer()
        lb.fit(self.y_test)
        self.y_test = lb.transform(self.y_test)
        y_pred = lb.transform(y_pred)
        auc_score= metrics.roc_auc_score(self.y_test,y_pred, average="macro")
        return auc_score


    @timing
    def fitter(self):
        """Function to perform actual fitting and prediction"""
        md= self.model.fit(self.X_train,self.y_train)
        train_score=md.score(self.X_train,self.y_train)
        y_pred =md.predict(self.X_test)
        accuracy=accuracy_score(self.y_test,y_pred)
        auc_score=self.multiclass_roc_auc_score(y_pred)
        return accuracy,auc_score,train_score,y_pred,md





