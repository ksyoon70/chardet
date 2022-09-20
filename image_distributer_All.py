"""
Created on 2022년 7월 31일
레이블 리스트를 읽어 레이블를 확인한다.
영상파일에서 레이블에 해당하는 폴더를 만들고 해당 폴더로 영상을 이동한다.
예) python image_distributer -l ./LPR_Labels1.txt -i ./image -o ./result -t ch -r [0.7,0.3]
-l label 파일
-i 이미지 위치
-o 결과 파일 저장 위치
-r train validation 분배 비율 default : 0.7,0.3 이다.
@author:  윤경섭
"""

import os,sys,shutil
import argparse
import pandas as pd
from label_tools import *
import random
import math

#------------------------------
# 수정할 내용
MIDDLE_PATH_NAME = 'datasets'
OUTPUT_FOLDER_NAME = 'out' # labelme로 출력할 디렉토리 이름 (현재 디렉토리 아래로 저장된다.)
DEFAULT_OBJ_TYPE = 'or'
DEFAULT_LABEL_FILE = "./LPR_Labels1.txt"  #라벨 파일이름
option_move = False # 원 파일을 옮길지 여부
#------------------------------
class_str = None   #클래스의 이름을 저장한다.

OBJECT_TYPES = ['ch','hr','vr','or']

for DEFAULT_OBJ_TYPE in OBJECT_TYPES :

    if DEFAULT_OBJ_TYPE == 'ch':        #문자 검사
        IMAGE_FOLDER_NAME = 'ch_images'
        class_str = "character"
    elif DEFAULT_OBJ_TYPE == 'hr':       #숫자검사
        IMAGE_FOLDER_NAME = 'hr_images'
        class_str = "hregion"
    elif DEFAULT_OBJ_TYPE == 'vr':     #지역문자 검사
        IMAGE_FOLDER_NAME = 'vr_images'
        class_str = "vregion"
    elif DEFAULT_OBJ_TYPE == 'or':       #v 지역문자 검사
        IMAGE_FOLDER_NAME = 'or_images'
        class_str = "oregion"
    elif DEFAULT_OBJ_TYPE == 'r6':     #h 지역문자 검사
        IMAGE_FOLDER_NAME = 'r6_images'
        class_str = "region6"
    elif DEFAULT_OBJ_TYPE == 'r' :       #r 지역문자 검사
        IMAGE_FOLDER_NAME = 'r_images'
        class_str = "region"
    elif DEFAULT_OBJ_TYPE == 'n':       #6 지역문자 검사
        IMAGE_FOLDER_NAME = 'n_images'
        class_str = "number"     
    else:
        print("{0} image folder name is Not supporeted".format(IMAGE_FOLDER_NAME))
        sys.exit(0)
    
    print('{}을 처리합니다.'.format(IMAGE_FOLDER_NAME))
        
    OUTPUT_FOLDER_NAME = class_str

    ROOT_DIR = os.path.dirname(__file__)
    DEFAULT_IMAGES_PATH = os.path.join(ROOT_DIR,MIDDLE_PATH_NAME,IMAGE_FOLDER_NAME)
    DEFAULT_OUPUT_PATH = os.path.join(ROOT_DIR,MIDDLE_PATH_NAME,OUTPUT_FOLDER_NAME)


    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="object split and save in jpeg and annotation files")

    parser.add_argument("-l",
                        "--labelfile",
                        help="Label file where the text files are stored.",
                        type=str,default=DEFAULT_LABEL_FILE)

    parser.add_argument("-i",
                        "--image_dir",
                        help="Path to the folder where the input image files are stored. ",
                        type=str, default=DEFAULT_IMAGES_PATH)
    # 출력 디렉토리를 설정ㅎㄴ다.
    parser.add_argument("-o",
                        "--output_dir",
                        help="Path of output images and jsons", type=str,default=DEFAULT_OUPUT_PATH)
    # 검색할 object type를 설정한다. 
    parser.add_argument("-t",
                        "--object_type",
                        help="object type ch : character n: number r: region", type=str,default=DEFAULT_OBJ_TYPE)
    # training / validateion  비율을 설정한다.
    parser.add_argument("-r",
                        "--ratio", type=float,
                        help="train validation ratio ex[0.7,0.3] ",default=[0.7,0.3], required=False)


    args = parser.parse_args()

    fLabels = pd.read_csv(args.labelfile, header = None )
    LABEL_FILE_CLASS = fLabels[0].values.tolist()
    LABEL_FILE_HUMAN_NAMES = fLabels[1].values.tolist()
    CLASS_DIC = dict(zip(LABEL_FILE_CLASS, LABEL_FILE_HUMAN_NAMES))

    #클래스를 각각 그룹별로 나눈다.
    CH_CLASS = LABEL_FILE_CLASS[21:111]  #문자열 클래스
    NUM_CLASS = LABEL_FILE_CLASS[11:21]  #숫자 클래스
    REGION_CLASS = LABEL_FILE_CLASS[111:-1] #지역문자 클래스
    VREGION_CLASS = LABEL_FILE_CLASS[111:128] #Vertical 지역문자 클래스
    HREGION_CLASS = LABEL_FILE_CLASS[128:145] #Horizontal 지역문자 클래스
    OREGION_CLASS = LABEL_FILE_CLASS[145:162] #Orange 지역문자 클래스
    REGION6_CLASS = LABEL_FILE_CLASS[162:-1] #6 지역문자 클래스

    #사람이 볼수있는 이름으로 나눈다.
    CH_HUMAN_NAMES = LABEL_FILE_HUMAN_NAMES[21:111]  #문자열 클래스
    NUM_HUMAN_NAMES = LABEL_FILE_HUMAN_NAMES[11:21]  #숫자 클래스
    REGION_HUMAN_NAMES = LABEL_FILE_HUMAN_NAMES[111:-1] #지역문자 클래스
    VREGION_HUMAN_NAMES = LABEL_FILE_HUMAN_NAMES[111:128] #Vertical 지역문자 클래스
    HREGION_HUMAN_NAMES = LABEL_FILE_HUMAN_NAMES[128:145] #Horizontal 지역문자 클래스
    OREGION_HUMAN_NAMES = LABEL_FILE_HUMAN_NAMES[145:162] #Orange 지역문자 클래스
    REGION6_HUMAN_NAMES = LABEL_FILE_HUMAN_NAMES[162:-1] #6 지역문자 클래스

    train_ratio = args.ratio[0]
    validation_ratio = 1.0 - args.ratio[0]

    class_label = [];
    human_names= [];

    if args.object_type == 'ch':        #문자 검사
        class_label = CH_CLASS
        human_names = CH_HUMAN_NAMES
    elif args.object_type == 'n':       #숫자검사
        class_label = NUM_CLASS
        human_names = NUM_HUMAN_NAMES
        print("{0} type is Not supporeted yet".format(args.object_type))
        sys.exit(0)
    elif args.object_type == 'r':       #지역문자 검사
        class_label = REGION_CLASS
        human_names = REGION_HUMAN_NAMES
    elif args.object_type == 'vr':       #v 지역문자 검사
        class_label = VREGION_CLASS
        human_names = VREGION_HUMAN_NAMES
    elif args.object_type == 'hr':       #h 지역문자 검사
        class_label = HREGION_CLASS
        human_names = HREGION_HUMAN_NAMES
    elif args.object_type == 'or':       #o 지역문자 검사
        class_label = OREGION_CLASS
        human_names = OREGION_HUMAN_NAMES
    elif args.object_type == 'r6':       #6 지역문자 검사
        class_label = REGION6_CLASS
        human_names = REGION6_HUMAN_NAMES      
    else:
        print("{0} type is Not supporeted".format(args.object_type))
        sys.exit(0)
        
        
    #이미지 폴더가 있는지 확인한다.

    if not os.path.exists(args.image_dir) :
        print("No images folder exists. check the folder :",args.image_dir)
        sys.exit(0)
    
    #기존 폴더 아래 있는 출력 폴더를 지운다.
    if os.path.exists(args.output_dir) :
        shutil.rmtree(args.output_dir) 

    if not os.path.exists(args.output_dir) :
        createFolder(args.output_dir)
    

    #클래스 디렉토리를 만든다.
    #파일이 없는데도 만들어야 하는지 모르겠지만, 일단 만드는 코드는 아래 내용이다.
    #확인 결과 없는 영상에 대해서는 class가 무의미하다.

    # for label in class_label :
    #     tlabel_dir = os.path.join(args.output_dir,'train',label)
    #     if not os.path.exists(tlabel_dir):
    #         createFolder(tlabel_dir)
    #     vlabel_dir = os.path.join(args.output_dir,'validation',label)
    #     if not os.path.exists(vlabel_dir):
    #         createFolder(vlabel_dir)


    # images 디렉토리에서 image 파일을 하나씩 읽어 들인다. 
    src_dir = args.image_dir
    dst_dir = args.output_dir

    print('{} 에 파일에서 영상을 읽습니다.'.format(src_dir ))
    print('{} 에 파일을 저장합니다.'.format(dst_dir ))

    if os.path.exists(args.image_dir):
        image_ext = ['jpg','JPG','png','PNG']
        files = [fn for fn in os.listdir(src_dir)
                    if any(fn.endswith(ext) for ext in image_ext)]
        sfiles = []  #source file list
        for file in files:
            label = file.split('_')[-1]
            label = label[0:-4]
            # english class label을 얻는다. 
            if label in human_names :
                sfiles.append(file)
            else :
                continue
            
        # sfile 을 랜덤하게 섞고 난 후 비율 대로 분리한다.
        fileLen = len(sfiles)
        if fileLen :
            print('처리할 파일 갯수 {0}'.format(fileLen))
            print('파일을 랜덤하게 섞습니다')
            random.shuffle(sfiles)
            train_file_count = int(fileLen*train_ratio)
            tfiles = sfiles[0:train_file_count]
            vfiles = sfiles[train_file_count :]
            print('train 파일갯수 : {0} validation 파일갯수 : {1}'.format(len(tfiles),len(vfiles)))
            if len(tfiles) :
                for tfile in tfiles:
                    label = tfile.split('_')[-1]
                    label = label[0:-4]
                    en_label = class_label[human_names.index(label)]
                    en_label_dir = os.path.join(dst_dir,'train',en_label)
                    if not os.path.exists(en_label_dir):
                        createFolder(en_label_dir)
                    #파일을 레이블 디렉토리로 복사한다.
                    src = os.path.join(src_dir,tfile)
                    dst = os.path.join(dst_dir,en_label_dir,tfile)
                    if option_move :
                        shutil.move(src,dst)
                    else :
                        shutil.copy(src,dst)
            if len(vfiles) :
                for vfile in vfiles:
                    label = vfile.split('_')[-1]
                    label = label[0:-4]
                    en_label = class_label[human_names.index(label)]
                    en_label_dir = os.path.join(dst_dir,'validation',en_label)
                    if not os.path.exists(en_label_dir):
                        createFolder(en_label_dir)
                    #파일을 레이블 디렉토리로 복사한다.
                    src = os.path.join(src_dir,vfile)
                    dst = os.path.join(dst_dir,en_label_dir,vfile)
                    if option_move :
                        shutil.move(src,dst)
                    else :
                        shutil.copy(src,dst)
        else :
            print('처리할 파일이 없습니다')
            
        #valid에는 있는데 train에는 없는 파일들을 train에 복사한다.
        #먼저 train의 하위 디렉토리를 구한다.
        dst_train_dir = os.listdir(os.path.join(dst_dir,'train'))
        #validation의 하위 디렉토리를 구한다.
        dst_valid_dir = os.listdir(os.path.join(dst_dir,'validation'))
        
        not_in_train_dir = [ dir_name for dir_name in dst_valid_dir if not dir_name  in dst_train_dir ] 
        # 없는 파일을 train으로 비율만큼 복사한다.
        
        for dir_name in not_in_train_dir :
            dir_path = os.path.join(dst_dir,'validation',dir_name)
            if not os.path.exists(dir_path) :
                createFolder(dir_path)
            file_len = len(os.listdir(dir_path))
            filelist = os.listdir(dir_path)     
            random.shuffle(filelist)
            copy_file_count = math.ceil(file_len*train_ratio)
            copy_file_list = filelist[0:copy_file_count]
            
            for file in copy_file_list:
                src = os.path.join(dst_dir,'validation',dir_name,file)
                dst = os.path.join(dst_dir,'train',dir_name,file)
                dst_directory = os.path.join(dst_dir,'train',dir_name)
                if not os.path.exists(dst_directory) :
                    createFolder(dst_directory)
                shutil.copy(src,dst)
        
        #반대 과정을 한다. 
        not_in_valid_dir = [ dir_name for dir_name in dst_train_dir if not dir_name  in dst_valid_dir]  
            
        for dir_name in not_in_valid_dir :
            dir_path = os.path.join(dst_dir,'train',dir_name)
            if not os.path.exists(dir_path) :
                createFolder(dir_path)
            file_len = len(os.listdir(dir_path))
            filelist = os.listdir(dir_path)           
            random.shuffle(filelist)
            copy_file_count = math.ceil(file_len*validation_ratio)
            copy_file_list = filelist[0:copy_file_count]
            
            for file in copy_file_list:
                src = os.path.join(dst_dir,'train',dir_name,file)
                dst = os.path.join(dst_dir,'validation',dir_name,file)
                dst_directory = os.path.join(dst_dir,'validation',dir_name)
                if not os.path.exists(dst_directory) :
                    createFolder(dst_directory)
                shutil.copy(src,dst)    
            

        print('처리완료')      
    else :
        print("Error! no json directory:",args.json_dir)       