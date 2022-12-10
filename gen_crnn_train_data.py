"""
CRNN용 train data를 생성한다.
@author:  윤경섭
"""

import os,sys
import argparse
import pandas as pd
import cv2
from label_tools import *
import matplotlib.pyplot as plt
from shutil import copyfile, move, rmtree
#------------------------------
# 수정할 내용

IMAGE_FOLDER_NAME = 'images' #이미지 파일에 있는 영상 파일이 있는 경로
CRNN_FOLDER = 'C://SPB_Data//CRNN-Keras//DB'  #복사할 폴더의 디렉토리 이름
OBJTYPES = ['ch','reg']
INPUT_FOLDDERS = ['ch_images','hr_images','vr_images','or_images']
OUTPUT_FOLDERS = ['char_train', 'reg_train','reg_train','reg_train']
#------------------------------
ROOT_DIR = os.path.dirname(__file__)

for index, input_folder in enumerate(INPUT_FOLDDERS) :
    if index == 0 or index == 1:
        dst_dir = os.path.join(CRNN_FOLDER,OUTPUT_FOLDERS[index])
        if os.path.os.path.exists(dst_dir):
            rmtree(dst_dir)
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
    src_dir = os.path.join(ROOT_DIR,'datasets',input_folder) 
    image_ext = ['jpg','JPG','png','PNG']
    image_list = [fn for fn in os.listdir(src_dir)
                if any(fn.endswith(ext) for ext in image_ext)]
    total_images_files = len(image_list)
    print('{} 폴더에서 {}폴더로 총 {}개 파일을 복사합니다.'.format(input_folder,OUTPUT_FOLDERS[index],total_images_files))
    for image_filename in image_list :
        src =  os.path.join(src_dir,image_filename)
        dst  = os.path.join(dst_dir,image_filename)
        copyfile(src, dst)