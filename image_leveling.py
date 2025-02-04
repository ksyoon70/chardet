#-*- encoding: utf8 -*-
"""
Created on 2025년 1월 27일
이 파일은 train 영역에 있는 데이터 중 각 class별로 갯수를 구하고,
평균보다 많은 이미지가 있으면 삭제한다. 
@author:  윤경섭
"""

import os, shutil
import sys
from pathlib import Path
from label_tools import *
import cv2
import random
#------------------------------------------------

#src_image_dir = r'E:\SPB_Data\chardet\datasets\character\train'
#src_image_dir = os.path.normpath(src_image_dir)
#src_image_dir = Path(src_image_dir)


def count_files_in_subfolders(directory):
    subfolder_counts = {}
    
    for root, dirs, files in os.walk(directory):
        if root == directory:  # 최상위 폴더는 스킵
            continue
        subfolder_name = os.path.basename(root)
        subfolder_counts[subfolder_name] = len(files)
    
    return subfolder_counts

def calculate_average_file_count(file_counts):
    if not file_counts:  # 하위 폴더가 없는 경우
        return 0
    total_files = sum(file_counts.values())
    average = total_files / len(file_counts)
    return average



# 메인 함수: 데이터 증폭 및 처리
def image_leveling(src_image_dir):
    # 경로 정리
    src_image_dir = Path(src_image_dir)
    if not src_image_dir.exists() or not src_image_dir.is_dir():
        print(f"Error: The provided directory '{src_image_dir}' does not exist.")
        return

    file_counts = count_files_in_subfolders(src_image_dir)

    # 파일 수 평균 계산
    average_count = calculate_average_file_count(file_counts)

    print(f"\n처리 전 평균 파일 수: {average_count:.2f}")
    print(f"\n폴더별 파일 갯수")
    # 결과 출력
    for folder, count in file_counts.items():
        print(f"{folder}: {count} files")

    total_del_file_count = 0
    # 평균보다 작은 폴더 augmentation
    for folder, count in file_counts.items():
        if count == 0:
            continue
        del_file_count = 0
        if count > int(average_count) :

            aug_folder =  os.path.join(src_image_dir,folder)
            image_ext = ['jpg','JPG','png','PNG']
            files = [fn for fn in os.listdir(aug_folder)
                        if any(fn.endswith(ext) for ext in image_ext)]

            # 랜덤하게 삭제할 파일 개수 계산
            files_to_delete_count = count - int(average_count)

            # 삭제할 파일 리스트 선택
            files_to_delete = random.sample(files, files_to_delete_count)

            for file in files_to_delete:
                file_path = os.path.join(aug_folder, file)
                try:
                    os.remove(file_path)
                    #print(f"Deleted: {file_path}")
                    total_del_file_count += 1
                    del_file_count += 1
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
    
    
    file_counts = count_files_in_subfolders(src_image_dir)

    # 파일 수 평균 계산
    post_average_count = calculate_average_file_count(file_counts)

    print(f"\n처리 후 평균 파일 수: {post_average_count:.2f}")
    print(f"\n폴더별 파일 갯수")
    # 결과 출력
    for folder, count in file_counts.items():
        print(f"{folder}: {count} files")

    # 전체 요약
    if total_del_file_count > 0:
        print('총 {}장의 증폭 이미지를 삭제제했습니다.'.format(total_del_file_count))
    else:
        print('증폭할 이미지가 없습니다.')

    return int(average_count)    
        
# ------------------------------------------------
# 단독 실행 여부 확인
if __name__ == "__main__":
    # 명령줄에서 인자를 받을 경우 처리
    if len(sys.argv) > 1:
        src_image_dir = sys.argv[1]
    else:
        # 기본 디렉토리
        src_image_dir = r'E:\SPB_Data\chardet\datasets\out1\character\train'
        src_image_dir = os.path.normpath(src_image_dir)
        src_image_dir = Path(src_image_dir)

    image_leveling(src_image_dir)