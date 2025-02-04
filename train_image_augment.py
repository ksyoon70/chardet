#-*- encoding: utf8 -*-
"""
Created on 2025년 1월 23일
이 파일은 train 영역에 있는 데이터 중 각 class별로 갯수를 구하고,
필요 시 데이터 증폭(Augmentation)을 수행합니다.
@author:  윤경섭
"""

import os, shutil
import sys
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from label_tools import *
import cv2
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


def apply_rotation(image, angle=30):
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def apply_width_shift(image, shift=0.1):
    h, w = image.shape[:2]
    shift_val = int(w * shift)
    matrix = np.float32([[1, 0, shift_val], [0, 1, 0]])
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def apply_height_shift(image, shift=0.1):
    h, w = image.shape[:2]
    shift_val = int(h * shift)
    matrix = np.float32([[1, 0, 0], [0, 1, shift_val]])
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def apply_shear(image, shear=0.1):
    h, w = image.shape[:2]
    matrix = np.float32([[1, shear, 0], [0, 1, 0]])
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def apply_zoom(image, zoom=0.8):
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    radius_x, radius_y = int(w * zoom // 2), int(h * zoom // 2)
    cropped = image[center_y - radius_y:center_y + radius_y, center_x - radius_x:center_x + radius_x]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def apply_brightness(image, factor=0.8):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)



# 메인 함수: 데이터 증폭 및 처리
def process_and_augment(src_image_dir, average_count = None ):
    # 경로 정리
    src_image_dir = Path(src_image_dir)
    if not src_image_dir.exists() or not src_image_dir.is_dir():
        print(f"Error: The provided directory '{src_image_dir}' does not exist.")
        return

    file_counts = count_files_in_subfolders(src_image_dir)

    # 파일 수 평균 계산
    if average_count is None:
        average_count = calculate_average_file_count(file_counts)

    print(f"\nAverage file count per subfolder: {average_count:.2f}")

    # 결과 출력
    for folder, count in file_counts.items():
        print(f"{folder}: {count} files")

    total_gen_file_count = 0
    # 평균보다 작은 폴더 augmentation
    for folder, count in file_counts.items():
        if count == 0:
            continue
        gen_file_count = 0
        if count < int(average_count) :
            if count + gen_file_count > int(average_count) : #생성 파일의 개수가 평균 이상이 되면 탈출
                continue
            aug_folder =  os.path.join(src_image_dir,folder)
            image_ext = ['jpg','JPG','png','PNG']
            files = [fn for fn in os.listdir(aug_folder)
                        if any(fn.endswith(ext) for ext in image_ext)]
            
            # 효과 리스트 정의
            transformations = {
                "rotation": apply_rotation,
                "width_shift": apply_width_shift,
                "height_shift": apply_height_shift,
                "shear": apply_shear,
                "zoom": apply_zoom,
                "brightness": apply_brightness,
            }

            # 회전 각도 범위: -30 ~ 30, 5도 간격
            rotation_angles = range(-30, 31, 5)
            # 밝기 범위: 0.2 ~ 1.0, 0.1 간격
            brightness_values = np.arange(0.2, 1.1, 0.1)  

            for file in files:
                label = file.split('_')[-1]
                label = label[0:-4]
                basename, ext = os.path.splitext(file)
                idx = basename.rfind('_')

                if idx != -1:  # '_'가 존재한다면
                    header_name = basename[:idx]
                else:
                    # '_'가 없으면 전체 파일명 그대로
                    header_name = basename

                img_path = os.path.join(aug_folder, file)
                img = imread(img_path)

                if img is None:
                    continue

                #동일 폴더에 저장한다.
                output_dir = aug_folder

                # -----------------------------
                # (1) rotation 범위 적용
                # -----------------------------
                for angle in rotation_angles:
                    # angle = 0 인 경우 기존 이미지와 동일하므로 제외하고 싶으면 continue
                    # if angle == 0:
                    #     continue
                    transformed_img = apply_rotation(img, angle)
                    save_path = os.path.join(
                        aug_folder,
                        f"{header_name}_AUG_rotation_{angle}_{label}{ext}"
                    )
                    imwrite(save_path, transformed_img)
                    gen_file_count += 1
                    total_gen_file_count += 1
                    
                    if count + gen_file_count > int(average_count):
                        break
                
                if count + gen_file_count > int(average_count):
                    break
                
                # -----------------------------
                # (2) brightness 범위 적용
                # -----------------------------
                for factor in brightness_values:
                    # factor = 1.0은 원본과 동일 -> 굳이 제외하고 싶다면 continue 처리
                    # if abs(factor - 1.0) < 1e-9:
                    #    continue
                    transformed_img = apply_brightness(img, factor)
                    save_path = os.path.join(
                        aug_folder,
                        f"{header_name}_AUG_brightness_{factor:.1f}_{label}{ext}"
                    )
                    imwrite(save_path, transformed_img)
                    gen_file_count += 1
                    total_gen_file_count += 1

                    if count + gen_file_count > int(average_count):
                        break
                
                if count + gen_file_count > int(average_count):
                    break
                
                # -----------------------------
                # (3) 그 외 변환들(고정 1회씩)
                # -----------------------------
                for key, transform_func in transformations.items():
                    transformed_img = transform_func(img)
                    save_path = os.path.join(
                        aug_folder,
                        f"{header_name}_AUG_{key}_{label}{ext}"
                    )
                    imwrite(save_path, transformed_img)
                    gen_file_count += 1
                    total_gen_file_count += 1

                    if count + gen_file_count > int(average_count):
                        break

            # 폴더별 augmentation 이후 요약
            if gen_file_count > 0:
                print(f"[{folder}] 폴더에서 {gen_file_count}장의 이미지를 생성했습니다.")

    # 전체 요약
    if total_gen_file_count > 0:
        print('총 {}장의 증폭 이미지를 생성했습니다.'.format(total_gen_file_count))
    else:
        print('증폭할 이미지가 없습니다.')
        
        
# ------------------------------------------------
# 단독 실행 여부 확인
if __name__ == "__main__":
    # 명령줄에서 인자를 받을 경우 처리
    if len(sys.argv) > 1:
        src_image_dir = sys.argv[1]
    else:
        # 기본 디렉토리
        src_image_dir = r'E:\SPB_Data\chardet\datasets\character\train'
        src_image_dir = os.path.normpath(src_image_dir)
        src_image_dir = Path(src_image_dir)

    process_and_augment(src_image_dir)