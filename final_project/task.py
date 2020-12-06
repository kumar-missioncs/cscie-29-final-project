
from final_project.classify import*
from final_project.data import*
import os
from luigi import*
import luigi
from luigi import ExternalTask, Parameter, Task, LocalTarget
from luigi.contrib.s3 import S3Target
import pandas as pd



class GetImage(ExternalTask):
    """ Class to setup an external task to give S3Target of hyperspectral image"""

    # my data bucket in S3.
    S3_ROOT =Parameter(default= "s3://dktproject/data/")


    def output(self):
        storage_options = dict( requester_pays=True )
        target = S3Target(f"{self.S3_ROOT}",format= luigi.format.Nop,
                          storage_options = storage_options)
        return target


class DownloadImage(Task):
    """Class to Download images"""
    S3_ROOT= Parameter(default= "s3://dktproject/data/")
    LOCAL_DIR = Parameter(default='/data/')
    image = Parameter(default = 'Indian_pines_corrected.mat') # Luigi parameter to get hyperspectral image
    gt = Parameter( default = 'Indian_pines_gt.mat') # get the ground truth of the hyperspectral image
    gt_names= Parameter(default='gt_names.csv')
    LOCAL_ROOT = os.getcwd()+'/data/'
    #print(LOCAL_ROOT)

    def requires(self):
        # Depends on the ContentImage ExternalTask being complete
        return {'image': self.clone(GetImage) }

    def output(self):
        LOCAL_PATH=os.getcwd()+self.LOCAL_DIR
        target = luigi.LocalTarget(f"{LOCAL_PATH}{self.image}")
        return target

    def run(self):
        LOCAL_PATH=os.getcwd()+self.LOCAL_DIR # I need to do this trick to facilitate pytest run
        print("====================================",LOCAL_PATH)
        # Read the hyperspectral image from the S3 write to local directory
        with S3Target(path = f'{self.S3_ROOT}{self.image}',format = luigi.format.Nop).open('r') as in_image:
            my_image = in_image.read()
        with LocalTarget(path = f'{LOCAL_PATH}{self.image}',format = luigi.format.Nop ).open('w') as out_image:
            out_image.write(my_image)
        # Read ground truth from S3 and write to local directory
        with S3Target(path = f'{self.S3_ROOT}{self.gt}',format = luigi.format.Nop).open('r') as in_gt:
            my_gt = in_gt.read()
        with LocalTarget(path = f'{LOCAL_PATH}{self.gt}',format = luigi.format.Nop ).open('w') as out_gt:
            out_gt.write(my_gt)
        # Read ground truth names from S3 and write to local directory
        with S3Target(path = f'{self.S3_ROOT}{self.gt_names}',format = luigi.format.Nop).open('r') as in_gt_names:
            my_gt_names = in_gt_names.read()
        with LocalTarget(path = f'{LOCAL_PATH}{self.gt_names}',format = luigi.format.Nop ).open('w') as out_gt_names:
            out_gt_names.write(my_gt_names)



class VisualizeData(Task):
    """ Class to read .mat files in dataframes and normalize it and print visuals"""
    #LOCAL_ROOT=os.getcwd()+'/data/'  # the raw image data resides here
    LOCAL_DIR = Parameter(default='/data/')
    image= Parameter(default = 'Indian_pines_corrected.mat') # main image file
    gt= Parameter(default = 'Indian_pines_gt.mat') # ground truth files
    IMAGE_KEY=Parameter(default= 'indian_pines_corrected') # dic key for the image
    GT_KEY=Parameter(default= 'indian_pines_gt') # dic key for the ground truth
    RESULT_DIR= Parameter(default='/results/') # the results resides here
    test_flag=Parameter(default="No") # A flag to redirect test files in the test data and results folders
    Spectra = 'Spectral_bands.png'
    NORM_DATA='image_norm.parquet'
    IMAGE_FILE_SUFFIX= Parameter(default="mat") # this allows to work with multiple types of files.
    
    r1 = Parameter(default="1") # the starting range of list of number
    r2 = Parameter(default="8") # end range of the list of number

    def requires(self):
        # Depends on the ContentImage ExternalTask being complete
        return {'image': self.clone(GetImage) }

    def visualize_normalize(self):
        LOCAL_PATH=os.getcwd()+self.LOCAL_DIR # I need to do this trick to facilitate pytest run
        RESULT_PATH=os.getcwd()+self.RESULT_DIR # I need to do this trick to facilitate pytest run
        # class need to get (data_path,result_path,image_file,gt_file,image_key,gt_key,r1,r2, data_file_suffix)
        p1=PlotNormData(LOCAL_PATH,RESULT_PATH,self.image,self.gt,self.IMAGE_KEY,self.GT_KEY,
                        self.r1,self.r2,self.IMAGE_FILE_SUFFIX,self.test_flag)
        p1.plot_gt_spectral()
        p1.normalize_save_data()


    def output(self):
        RESULT_PATH=os.getcwd()+self.RESULT_DIR  # I need to do this trick to facilitate pytest run
        # Return the parquet file for the next stage of processing.
        target = luigi.LocalTarget(f"{RESULT_PATH}{self.NORM_DATA}")
        return target

    def run(self):
        self.visualize_normalize()



class ClassifyImage(Task):
    """The class will normalize the data and perform principle components analysis """
    normData=Parameter('image_norm.parquet')
    RESULT_DIR= Parameter(default='/results/') # the results resides here
    dim_x=Parameter(default = "145") # These are the X,Y dimension of the spectral image
    dim_y=Parameter(default = "145")
    LOCAL_DIR = Parameter(default='/data/')
    gt_names=Parameter(default='gt_names.csv') # the file where ground truth resides
    test_flag=Parameter(default="No")

    def requires(self):
        # Depends on the VisulaizeData to get completed
        return { 'normData': self.clone(VisualizeData) }



    def classifer_pipeline(self):
        """Function to get the multi classifiers pipelined"""
        LOCAL_PATH=os.getcwd()+self.LOCAL_DIR # I need to do this trick to facilitate pytest run
        RESULT_PATH=os.getcwd()+self.RESULT_DIR # I need to do this trick to facilitate pytest run
        df=pd.read_parquet(f'{RESULT_PATH}{self.normData}',engine='fastparquet')
        dfnames=pd.read_csv(f'{LOCAL_PATH}{self.gt_names}')
        names=list(dfnames.iloc[:,1].values)
        print(names)

        #Split dataset and perform PCA
        s1= SplitafterPCA(df,num_components =20 , test_size =0.15,random_state=123)
        df_pca, X_train, X_test, y_train, y_test=s1.split_train_test()


        # parameter for the  classifiers
        rf_param={'n_estimators':80,'max_depth':15,'max_features':8 }
        svc_param={'C':100,'kernel':'rbf','cache_size':10240 }
        models_dict={"RF":rf_param,"SVC":svc_param}
        for item in models_dict.items():
            clf,clf_name=Classifers(item[0],item[1]).create_model()
            print("**************************************************\n")
            print("Classifier used is ", clf)
            accuracy,auc_score,train_score,y_pred,md = Fit_predict(clf,X_train,y_train,X_test, y_test).fitter()
            print("Classification accuracy is\n",accuracy)
            print("AUC score for classification is\n",auc_score)
            image_shape= [int(self.dim_x), int(self.dim_y)]
            # get results after classification
            pr= Printresults(md,clf_name,y_test,y_pred,df_pca,names,image_shape,RESULT_PATH,self.test_flag)
            pr.print_reports()

    def run(self):
        self.classifer_pipeline()








