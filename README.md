# Final Project

### This repo is a snapshot of the my final project repo hosted within cscie-29 organization. As such this can be used to just review the materials done so far. But it does not reflect real commit history and can not be used for any grading and evaluation purpose. This is solely created to let my peers view the work I have done so far and provide any feedback they think will help this work. Thanks for visiting this project.


[![Build Status](https://travis-ci.com/csci-e-29/2020fa-final-project-kumar-missioncs.svg?token=U6Mg2kwRvGjaRy2Ko3SB&branch=master)](https://travis-ci.com/csci-e-29/2020fa-final-project-kumar-missioncs)

[![Maintainability](https://api.codeclimate.com/v1/badges/61750c78d69d8f3a7826/maintainability)](https://codeclimate.com/repos/5fcafd0ff27b3c018e006aff/maintainability)

[![Test Coverage](https://api.codeclimate.com/v1/badges/61750c78d69d8f3a7826/test_coverage)](https://codeclimate.com/repos/5fcafd0ff27b3c018e006aff/test_coverage)
## Preface

This repo is designed as the final project for the CSCI-29 Fall 2020. The objective of this project is to
use the Adavance python techniques to create a hyperspectral image classification pipeline.

### HyperSpectral imaging
Hyperspectral imaging, like other spectral imaging, collects and processes information from across the electromagnetic spectrum.[1] The goal of hyperspectral imaging is to obtain
the spectrum for each pixel in the image of a scene, with the purpose of finding objects, identifying materials, or
detecting processes. The source is taken from
https://en.wikipedia.org/wiki/Hyperspectral_imaging

![alt text](http://lesun.weebly.com/uploads/2/6/7/2/26724130/6095996_orig.jpg)


![alt text](http://large.stanford.edu/courses/2015/ph240/islam1/images/f1.png)

### Dataset
The hyperspectral datasets used for this project are pubilically available from following two websites:
http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
http://lesun.weebly.com/hyperspectral-data-set.html

### Project modules
In the final_project folder there are three modules, which have dedicated functions:
#### 1. data.py:
This will be doing data preprocssing for the multiple types of files at present only .mat and .npy files are supported.
This module also normalize data and save in .parquet and .csv format for the further processing.

#### 2.Classify.py:
This is basically performing PCA, spplit test-train dataset and classification. At present it supports only Random Forest
and Support Vector classifier.

#### 3. task.py:
This module is a full luigi pipeline for running all tasks from begining to end. Data is taken from S3 storage and
processed, normalized, reduced in dimension and final results are printed in csv, and png files. Each result file is
time-stamped.
