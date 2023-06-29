# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 14:42:45 2022
h5로 저장한 모델을 tensorflow savemodel로 컨버전 한다.
@author: headway
"""
import numpy as np
import os, shutil, sys
import tensorflow as tf
import matplotlib.pyplot as plt
# 한글 폰트 사용을 위해서 세팅
from matplotlib import font_manager, rc

from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import load_model
import natsort
import time
import pandas as pd
import argparse

font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.rcParams['axes.unicode_minus'] = False  ## 추가해줍니다. 

#GPU 사용시 풀어 놓을 것
"""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
"""

#----------------------------
DEFAULT_OBJ_TYPE = 'ch'
MODEL_FILE_NAME ='character_resnet50_20230120-031916_model_epoch_44_val_acc_0.9754.h5'
WEIGHT_FILE_NAME = 'character_resnet50_20230119-234527_weights_epoch_034_val_acc_0.979.h5'
#----------------------------


# Initiate argument parser
parser = argparse.ArgumentParser(
    description="object split and save in jpeg and annotation files")
parser.add_argument("-m",
                    "--modelfile",
                    help="Label model file where the text files are stored.",
                    type=str,default=MODEL_FILE_NAME)
parser.add_argument("-w",
                    "--weightlfile",
                    help="Label weight file where the text files are stored.",
                    type=str,default=WEIGHT_FILE_NAME)


args = parser.parse_args()


    
trained_dir = './trained'
if not os.path.isdir(trained_dir):
    os.mkdir(trained_dir)


    
#read model
model = load_model(MODEL_FILE_NAME)
#read weight value from trained dir
weight_path = os.path.join(trained_dir,WEIGHT_FILE_NAME)
model.load_weights(weight_path)

model.save('ch_model')

print('Tensorflow SaveModel 로 저장했습니다')
        





