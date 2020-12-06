import os
from luigi import*
import pandas as pd
from scipy.io import loadmat
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from csci_utils.io.io import atomic_write


class CreateList:
    """Class to create a list of numbers on demand"""
    def __init__(self,r1,r2):
        self.r1=r1
        self.r2=r2

    # Program to Create list  with integers within given range
    # I need to do this because of some strange error caused by range() function
    def createfilelist(self):
        # Testing if range r1 and r2 are equal
        r1=int(self.r1)
        r2=int(self.r2)
        if (r1 == r2):
            return r1
        else:
            # Create empty list to store file numbers
            res = []
            while(r1 < r2+1 ):
                res.append(r1)
                r1 += 1
            return res

class PreprocessMatData:
    """ Class to preprocess .mat type file and convert in numpy array"""

    def __init__(self,data_path,result_path,image_file,gt_file,image_key,gt_key):
        self.data_path=data_path
        self.result_path=result_path
        self.image_file= image_file
        self.gt_file=gt_file
        self.image_key=image_key
        self.gt_key=gt_key


    def get_data(self):
        """  read the data and ground truth from given .mat files
        :return: X: numpy array of image, y: numpy array of ground truth
        """
        dataset_file = loadmat(f'{self.data_path}{self.image_file}') # this will results in the dictionary dictionary for image
        gt_file = loadmat(f'{self.data_path}{self.gt_file}') # this will results in the dictionary for GT
        arrIP = dataset_file[self.image_key] # from the given key take just 3D image value
        gt = gt_file[self.gt_key] # from Key get the GT values
        return arrIP,gt



class PreprocessNumpyData:
    """Class to load npy type file"""
    def __init__(self,data_path,result_path,image_file,gt_file):
        self.data_path=data_path
        self.result_path=result_path
        self.image_file= image_file
        self.gt_file=gt_file


    def get_data(self):
        """  read the data and ground truth from given files
        :return: X: numpy array of image, y: numpy array of ground truth
        """
        arrIP = np.load(f'{self.data_path}{self.image_file}') # this will results in the dictionary dictionary for image
        gt = np.load(f'{self.data_path}{self.gt_file}') # this will results in the dictionary for GT
        return arrIP,gt

### Note: Technically it is possible to merge PreprocessNumpyData and PreprocessMatData classes together
### for this set of problem. However I am keeping it seperate in a vision that various hyperspectral image files type
### have very different requirements to preporcess and convert them in the numpy type arrays.



class PlotNormData:
    """Class which plots EDA results of the dataset"""

    def __init__(self,data_path,result_path,image_file,gt_file,image_key,gt_key,r1,r2, data_file_suffix,test_flag):
        self.result_path=result_path # path where the results are stored
        self.suffix=data_file_suffix #path where suffix of the file is stored
        self.data_path=data_path # path to stored data for processing
        self.image_file= image_file  # main hyperspectral image file
        self.gt_file=gt_file  # main ground truth file
        self.image_key=image_key # main image key
        self.gt_key=gt_key # GT key
        self.r1=r1
        self.r2=r2
        self.test_flag=test_flag
        self.mat=PreprocessMatData(self.data_path,self.result_path,self.image_file,
                                   self.gt_file,self.image_key,self.gt_key)  # composing the classes for the .mat Array type data
        self.nmpy=PreprocessNumpyData(self.data_path,self.result_path,self.image_file,self.gt_file) # composing the classes for the Numpy Array type data
        self.lis =CreateList(self.r1,self.r2) # compose list creation class




    def plot_gt_spectral(self):
       """ Function to plot the ground truth and spectral band"""
       if self.suffix == "mat":
            arrIP,gt = self.mat.get_data() # get data from mat type file
       elif self.suffix == "npy":
            arrIP,gt = self.nmpy.get_data() # get data from mat type file
       else:
            print("Unsupported file type")
            raise AttributeError()

        #plotting the groud truth to show where different vegitations are situated.

       fig,ax =plt.subplots(constrained_layout=True)
       cax = ax.contourf(gt, cmap='nipy_spectral')
       cbar = fig.colorbar(cax)
       plt.axis('off')
       plt.savefig(f'{self.result_path}Image_GT.png')
       # Visualize randomly selected eigth spectral bands out of 200 spectral bands
       fig = plt.figure(figsize = (12, 6))
       # here the default r1 and r2 values are used.
       list1= self.lis.createfilelist()
       for i in list1:
            fig.add_subplot(2,4, i)
            q = np.random.randint(arrIP.shape[2])
            plt.imshow(arrIP[:,:,q], cmap='nipy_spectral')
            plt.axis('off')
            plt.title(f'Spectral Band - {q}')

       plt.savefig(f'{self.result_path}Spectral_bands.png')



    def normalize_save_data(self):
        """Function to normalize and save the data in CSV file"""
        #print("===============================================\n")
        if self.suffix == "mat":
            arrIP,gt = self.mat.get_data() # get data from mat type file
        elif self.suffix == "npy":
            arrIP,gt = self.nmpy.get_data() # get data from mat type file
        else:
            print("Unsupported file type")
            raise AttributeError()
        #print("===================+++++++++================\n")

        # reshaping the data for the classification
        X = np.reshape( arrIP, (arrIP.shape[0]*arrIP.shape[1],arrIP.shape[2]))

        # Normalisation of data
        normalized_X =  preprocessing.normalize(X)

        # converting numpy array to dataframe
        # please note that I am using normalized data to create the Data Frame which will be used
        # for further processing.
        df = pd.DataFrame(data = normalized_X)
        df_class =pd.DataFrame(data = gt.ravel())
        df = pd.concat([df, df_class], axis =1)

        #Override the default initialized values of r1 and r2 using dict of object
        self.lis.__dict__['r1']=1
        self.lis.__dict__['r2']=arrIP.shape[2] # get the number of spectral bands
        list3=self.lis.createfilelist()
        # set the names of the columns in the dataset
        df.columns = [f'band{i}' for i in list3]+['classes']
        # filename=f'{self.result_path}image_norm.parquet' # worked with cli
        # df.to_parquet(filename,engine='fastparquet',compression=None) #
        # filename2=f'{self.result_path}image_norm.csv'
        # df.to_csv(filename2)

        filename=f'{self.result_path}image_norm.parquet' # worked with cli
        if os.path.exists(filename) == False :
        # atomically write parquet file by getting the path where to write it
            with atomic_write(filename, "w", False) as f:
                dir_path = f
            if self.test_flag == "No":
                new_file= '{0}/results/image_norm.parquet'.format(str(dir_path)) # worked with cli
            else:
                new_file= '{0}/test/results/image_norm.parquet'.format(str(dir_path)) # worked with pytest
            df.to_parquet(new_file,engine='fastparquet',compression=None)

        filename2=f'{self.result_path}image_norm.csv'
        if os.path.exists(filename2) == False :
        # atomically write csv file by getting the path where to write it
            with atomic_write(filename2, "w", False) as f:
                dir_path = f
            if self.test_flag=="No":
                new_file2= '{0}/results/image_norm.csv'.format(str(dir_path))
            else:
                new_file2= '{0}/test/results/image_norm.csv'.format(str(dir_path))
            df.to_csv(new_file2)
