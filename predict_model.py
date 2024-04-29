# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 14:42:45 2022

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
DEFAULT_LABEL_FILE = "./LPR_Labels1.txt"  #라벨 파일이름
IMG_SIZE = 224
THRESH_HOLD = 0.4
show_images = True
file_move = False
TEST_DIR_NAME = 'test'
RESULT_DIR_NAME = 'result'
DEFAULT_OBJ_TYPE = 'or'
MODEL_FILE_NAME ='oregion_resnet50_20240429-161754_model_epoch_22_val_acc_0.9701.h5'
WEIGHT_FILE_NAME = 'oregion_resnet50_20240429-161104_weights_epoch_016_val_acc_0.9810.h5'
#----------------------------

categories = []
result_cateories = []
dst_dirs = []

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="object split and save in jpeg and annotation files")
parser.add_argument("-l",
                    "--labelfile",
                    help="Label file where the text files are stored.",
                    type=str,default=DEFAULT_LABEL_FILE)

args = parser.parse_args()

fLabels = pd.read_csv(args.labelfile, header = None )
LABEL_FILE_CLASS = fLabels[0].values.tolist()
LABEL_FILE_HUMAN_NAMES = fLabels[1].values.tolist()
CLASS_DIC = dict(zip(LABEL_FILE_CLASS, LABEL_FILE_HUMAN_NAMES))

if DEFAULT_OBJ_TYPE == 'ch':        #문자 검사
    class_str = "character"
    model_dir = 'char_model'
    categorie_filename = 'character_categories.txt'
elif DEFAULT_OBJ_TYPE == 'n':       #숫자검사
    class_str = "number"
    model_dir = 'n_model'
    categorie_filename = 'number_categories.txt'
    print("{0} type is Not supporeted yet".format(args.object_type))
    sys.exit(0)
elif DEFAULT_OBJ_TYPE == 'r':       #지역문자 검사
    class_str = "region"
    model_dir = 'r_model'
    categorie_filename = 'region_categories.txt'
elif DEFAULT_OBJ_TYPE == 'vr':       #v 지역문자 검사
    class_str = "vregion"
    model_dir = 'vreg_model'
    categorie_filename = 'vregion_categories.txt'
elif DEFAULT_OBJ_TYPE == 'hr':       #h 지역문자 검사
    class_str = "hregion"
    model_dir = 'hreg_model'
    categorie_filename = 'hregion_categories.txt'
elif DEFAULT_OBJ_TYPE == 'or':       #o 지역문자 검사
    class_str = "oregion"
    model_dir = 'oreg_model'
    BATCH_SIZE = 4 # 갯수가 작아서 에러가 날수 있으므로...
    categorie_filename = 'oregion_categories.txt'
elif DEFAULT_OBJ_TYPE == 'r6':       #6 지역문자 검사
    class_str = "region6"
    model_dir = 'reg6_model'
    categorie_filename = 'region6_categories.txt'
else:
    print("{0} type is Not supported".format(args.object_type))
    class_str = "Not supported"
    sys.exit(0)

catLabels = pd.read_csv(categorie_filename, header = None )
categories = catLabels[0].values.tolist()
categories.append('no_categorie')

base_dir = './datasets'

if not os.path.isdir(base_dir):
    os.mkdir(base_dir)

#시험 폴더 위치 지정
src_dir = os.path.join(base_dir,class_str,TEST_DIR_NAME)
if not os.path.isdir(src_dir):
    os.mkdir(src_dir)
    
trained_dir = './trained'
if not os.path.isdir(trained_dir):
    os.mkdir(trained_dir)

#훈련 폴더 생성  
train_dir = os.path.join(base_dir,'train')
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)
    

#훈련 폴더에서 카테고리 취득
categorie_list = os.listdir(train_dir)
categorie_list = natsort.natsorted(categorie_list)
for categorie in categorie_list:
    categories.append(categorie)

categories.append('no_categorie')

cat_len = len(categories)    
    
#결과 저장 폴더 생성    
result_dir = os.path.join(base_dir,RESULT_DIR_NAME)
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

#결과 폴더 아래 카테고리 디렉토리 생성
for categorie in categories:
    dst_dir = os.path.join(result_dir,categorie)
    if not os.path.isdir(dst_dir):
         os.mkdir(dst_dir)
    dst_dirs.append(dst_dir)
    
#read model
model = load_model(MODEL_FILE_NAME)
#read weight value from trained dir
weight_path = os.path.join(trained_dir,WEIGHT_FILE_NAME)
model.load_weights(weight_path)


total_test_files = len(os.listdir(src_dir))

print('테스트용 이미지 갯수:',total_test_files)

recog_count = 0
fail_count = 0
false_recog_count = 0  #오인식 카운트
true_recog_count = 0

tilestr = None

start_time = time.time() # strat time

if len(os.listdir(src_dir)):

    files = os.listdir(src_dir)

    for file in files:
        
        try:
            img_path = os.path.join(src_dir,file)
            img = image.load_img(img_path,target_size=(IMG_SIZE,IMG_SIZE))
            img_tensor = image.img_to_array(img)
            img_tensor = np.expand_dims(img_tensor,axis=0)
            #if show_images :
            img_data = img_tensor/255.
            
            preds = model.predict(img_tensor)
            
            index = np.argmax(preds[0],0)
            
            src = os.path.join(src_dir,file)
           
            gtrue_label = file.split('_')[-1]
            gtrue_label = gtrue_label[0:-4]
            
            if preds[0][index] > THRESH_HOLD :
                predic_label = CLASS_DIC[categories[index]]
                tilestr = 'predict:{} GT:{}'.format(CLASS_DIC[categories[index]],gtrue_label) + '' + '  probability: {:.2f}'.format(preds[0][index]*100 )  + ' %'
                dst = os.path.join(dst_dirs[index],file)
                recog_count += 1
                if(gtrue_label == predic_label) :
                    true_recog_count += 1
                else:
                    false_recog_count += 1
            else:
                tilestr = 'Not sure but:{}GT:{}'.format(CLASS_DIC[categories[index]],gtrue_label) + '' + '  probability: {:.2f}'.format(preds[0][index]*100)  + ' %'
                dst = os.path.join(dst_dirs[cat_len -1 ],file)
                fail_count += 1
            #결과 디렉토리에 파일 저장
            if file_move :
                shutil.move(src,dst)
            else:
                shutil.copy(src,dst)
                
            if show_images :
                plt.title(tilestr)
                plt.imshow(img_data[0])
                plt.show()
        except Exception as e:
            pass
        
end_time = time.time()        
print("수행시간: {:.2f}".format(end_time - start_time))
print("총샘플수: {}".format(total_test_files))
print("건당 수행시간 : {:.2f}".format((end_time - start_time)/total_test_files))             
print('인식률: {:}/{}'.format(recog_count,total_test_files) +'  ({:.2f})'.format(recog_count*100/total_test_files) + ' %')
print('인식한것중 정인식: {:}/{}'.format(true_recog_count,recog_count) +'  ({:.2f})'.format(true_recog_count*100/recog_count) + ' %')
print('인식한것중 오인식: {:}/{}'.format(false_recog_count,recog_count) +'  ({:.2f})'.format(false_recog_count*100/recog_count) + ' %')
print('인식실패: {}/{}'.format(fail_count,total_test_files) +'  ({:.2f})'.format(fail_count*100/total_test_files) + ' %')
print('전체샘플중 정인식률: {}/{}'.format(true_recog_count,total_test_files) +'  ({:.2f})'.format(true_recog_count*100/total_test_files) + ' %')    
        
        





