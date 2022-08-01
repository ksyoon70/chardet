"""
Created on 2022년 7월 31일
레이블 리스트를 읽어 레이블를 확인한다.
영상파일에서 레이블에 해당하는 폴더를 만들고 해당 폴더로 영상을 이동한다.
예) python image_distributer -l ./LPR_Labels1.txt -i ./image -o ./result
-l label 파일
-i 이미지 위치
-o 결과 파일 저장 위치
@author:  윤경섭
"""

import os,sys,shutil
import argparse
import pandas as pd
from label_tools import *
#------------------------------
# 수정할 내용
MIDDLE_PATH_NAME = 'datasets'
OUTPUT_FOLDER_NAME = 'out' # labelme로 출력할 디렉토리 이름 (현재 디렉토리 아래로 저장된다.)
IMAGE_FOLDER_NAME = 'images' #이미지 파일에 있는 영상 파일이 있는 경로
DEFAULT_LABEL_FILE = "./LPR_Labels1.txt"  #라벨 파일이름
#------------------------------

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
                    help="object type ch : character n: number r: region", type=str,default='ch')
args = parser.parse_args()

fLabels = pd.read_csv(args.labelfile, header = None )
CLASS_NAMES = fLabels[0].values.tolist()
HUMAN_NAMES = fLabels[1].values.tolist()
CLASS_DIC = dict(zip(CLASS_NAMES, HUMAN_NAMES))

#클래스를 각각 그룹별로 나눈다.
CH_CLASS = CLASS_NAMES[21:111]  #문자열 클래스
NUM_CLASS = CLASS_NAMES[11:21]  #숫자 클래스
REGION_CLASS = CLASS_NAMES[111:-1] #지역문자 클래스
VREGION_CLASS = CLASS_NAMES[111:128] #Vertical 지역문자 클래스
HREGION_CLASS = CLASS_NAMES[128:145] #Horizontal 지역문자 클래스
OREGION_CLASS = CLASS_NAMES[145:162] #Orange 지역문자 클래스
REGION6_CLASS = CLASS_NAMES[162:-1] #6 지역문자 클래스



check_class = [];

if args.object_type == 'ch':        #문자 검사
    check_class = CH_CLASS
elif args.object_type == 'n':       #숫자검사
    check_class = NUM_CLASS
    print("{0} type is Not supporeted yet".format(args.object_type))
    sys.exit(0)
elif args.object_type == 'r':       #지역문자 검사
    check_class = REGION_CLASS
elif args.object_type == 'vr':       #v 지역문자 검사
    check_class = VREGION_CLASS
elif args.object_type == 'hr':       #h 지역문자 검사
    check_class = HREGION_CLASS
elif args.object_type == 'or':       #o 지역문자 검사
    check_class = OREGION_CLASS
elif args.object_type == 'r6':       #6 지역문자 검사
    check_class = REGION6_CLASS      
else:
    print("{0} type is Not supporeted".format(args.object_type))
    sys.exit(0)
    
    
#이미지 폴더가 있는지 확인한다.

if not os.path.exists(args.image_dir) :
    print("No images folder exists. check the folder :",args.image_dir)
    sys.exit(0)
    

if not os.path.exists(args.output_dir) :
    createFolder(args.output_dir)
 

#클래스 디렉토리를 만든다.
"""
for label in check_class :
    label_dir = os.path.join(args.output_dir,label)
    if not os.path.exists(label_dir):
        createFolder(label_dir)
"""

# images 디렉토리에서 image 파일을 하나씩 읽어 들인다. 
src_dir = args.image_dir
dst_dir = args.output_dir
if os.path.exists(args.image_dir):
    image_ext = ['jpg','JPG','png','PNG']
    files = [fn for fn in os.listdir(src_dir)
                  if any(fn.endswith(ext) for ext in image_ext)]
    for file in files:
        label = file.split('_')[1]
        label = label[0:-4]
        label_dir_name = CLASS_NAMES[HUMAN_NAMES.index(label)]
        label_dir = os.path.join(dst_dir,label_dir_name)
        if not os.path.exists(label_dir):
            createFolder(label_dir)
        #파일을 레이블 디렉토리로 복사한다.
        src = os.path.join(src_dir,file)
        dst = os.path.join(dst_dir,label_dir,file)
        shutil.move(src,dst)

    print('처리완료')      
else :
    print("Error! no json directory:",args.json_dir)       