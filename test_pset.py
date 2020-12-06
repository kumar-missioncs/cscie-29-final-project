#!/usr/bin/env python
# -*- coding: utf-8 -*-

from final_project.task import*
from final_project.classify import*
from final_project.data import*
from csci_utils.io.io import atomic_write
import shutil
import os
from luigi import *
import boto3
import botocore
from unittest import TestCase
import glob


class FakeFileFailure( IOError ):
    pass

class TestBucketExist(TestCase):
    """ Class to test the existence of bucket in S3"""
    def test_bucket(self):
        s3 = boto3.resource('s3')
        bucket_name = 'dktproject'
        bucket = s3.Bucket(bucket_name)
        try:
            s3.meta.client.head_bucket(Bucket=bucket.name)
            print("Bucket Exists!")
            return True
        except botocore.exceptions.ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 403:
                print("Private Bucket. Forbidden Access!")
                return True
            elif error_code == 404:
                print("Bucket Does Not Exist!")
                return False


class DownloadTest(TestCase):
    """ Class to test Download of hyperspectral data from S3"""
    # test data is coming from http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
    # Size of test data image file is  a 3-D vector of  86*63*204
    r1 = "1"
    r2 = "8"
    LOCAL_DIR = "/test/data/"
    S3_ROOT= "s3://dktproject/test/"
    image = 'SalinasA_corrected.mat' # Luigi parameter to get test hyperspectral image
    gt = 'SalinasA_gt.mat'   # test file ground truth of the hyperspectral image
    gt_names='gt_names.csv' # test file


    def test_download_image_model(self):
        parent_dir=os.getcwd()+"/test/"
        result_dir="results"
        data_dir="data"
        data_path=os.path.join(parent_dir,data_dir)
        results_path=os.path.join(parent_dir,result_dir)
        os.makedirs(data_path)
        os.makedirs(results_path)
        LOCAL_PATH=os.getcwd()+self.LOCAL_DIR
        build(
        {   GetImage(S3_ROOT=self.S3_ROOT),
            DownloadImage(LOCAL_DIR = self.LOCAL_DIR,S3_ROOT=self.S3_ROOT,
                          image=self.image,gt=self.gt,gt_names=self.gt_names),
        }, local_scheduler=True
        )
        # check that image file exist in data folder after getting from S3
        assert os.path.exists(LOCAL_PATH+ self.image)
        assert os.path.exists(LOCAL_PATH + self.gt)
        assert os.path.exists(LOCAL_PATH+ self.gt_names)


class VisualizeDataTest(TestCase):
    """ Class to read .mat files in dataframes and normalize it and print visuals"""
    #LOCAL_ROOT=os.getcwd()+'/data/'  # the raw image data resides here
    LOCAL_DIR = '/test/data/'
    image='SalinasA_corrected.mat' # main image file
    gt= 'SalinasA_gt.mat' # ground truth files
    IMAGE_KEY='salinasA_corrected' # dic key for the image
    GT_KEY='salinasA_gt' # dic key for the ground truth
    RESULT_DIR= '/test/results/' # the results resides here
    IMAGE_FILE_SUFFIX= "mat" # this allows to work with multiple types of files.
    test_flag="Yes"  # With yes value now all test data and results will be redirected in test Directory tree
    r1 = "1" # the starting range of list of number
    r2 = "8" # end range of the list of number
    def test_visualization(self):
        build(
        {   VisualizeData(LOCAL_DIR = self.LOCAL_DIR,image=self.image,gt=self.gt,IMAGE_KEY=self.IMAGE_KEY,
                          GT_KEY=self.IMAGE_KEY, RESULT_DIR=self.RESULT_DIR,IMAGE_FILE_SUFFIX=self.IMAGE_FILE_SUFFIX,
                          r1=self.r1,r2=self.r2,test_flag=self.test_flag),
        }, local_scheduler=True
        )
    def test_plot_norm(self):
        LOCAL_PATH=os.getcwd()+self.LOCAL_DIR # I need to do this trick to facilitate pytest run
        RESULT_PATH=os.getcwd()+self.RESULT_DIR # I need to do this trick to facilitate pytest run
        p1=PlotNormData(LOCAL_PATH,RESULT_PATH,self.image,self.gt,self.IMAGE_KEY,self.GT_KEY,
                        self.r1,self.r2,self.IMAGE_FILE_SUFFIX,self.test_flag)
        p1.plot_gt_spectral()
        p1.normalize_save_data()
        # check that image file and csv and parquet file exist in results folder
        assert os.path.exists(RESULT_PATH+ 'Image_GT.png')
        assert os.path.exists(RESULT_PATH + "image_norm.csv")
        assert os.path.exists(RESULT_PATH+ "image_norm.parquet")
        assert os.path.exists(RESULT_PATH+ "Spectral_bands.png")

class ClassiferTest(TestCase):

    normData='image_norm.parquet'
    RESULT_DIR= '/test/results/' # the results resides here
    dim_x="86" # These are the X,Y dimension of the spectral image
    dim_y="83"
    LOCAL_DIR = '/test/data/'
    gt_names='gt_names.csv' # the file where ground truth resides
    test_flag="Yes"
    def test_classificationLuigi(self):
        LOCAL_PATH=os.getcwd()+self.LOCAL_DIR # I need to do this trick to facilitate pytest run
        RESULT_PATH=os.getcwd()+self.RESULT_DIR # I need to do this trick to facilitate pytest run
        build(
        {   ClassifyImage(LOCAL_DIR = self.LOCAL_DIR,dim_x=self.dim_x,dim_y=self.dim_y,gt_names=self.gt_names,
                          RESULT_DIR=self.RESULT_DIR,normData=self.normData,test_flag=self.test_flag),
        }, local_scheduler=True
        )
        num_csv=len(glob.glob('./*.csv')) # count number of csv files
        num_png=len(glob.glob('./*.png')) # count number of png files
        self.assertIsNotNone(num_csv) # test that they are generated
        self.assertIsNotNone(num_png) # test that they are generated
        # remove the temporary results and data folder and generated files in test folder
        # comment these two lines if you want to see the generated test files
        shutil.rmtree(RESULT_PATH)
        shutil.rmtree(LOCAL_PATH)









