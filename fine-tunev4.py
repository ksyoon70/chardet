# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 11:37:06 2025

@author: headway
"""

import os,sys
from tkinter.messagebox import NO
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Activation, Flatten
from tensorflow.keras.layers import Reshape, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from Model import *
from parameter import *
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import argparse
import pandas as pd
import natsort
import cv2
import shutil
from PIL import Image
from pathlib import Path
from label_tools import *

import imgaug.augmenters as iaa  ### 추가
from tensorflow.keras.callbacks import ReduceLROnPlateau
K.set_learning_phase(0)
from tensorflow_addons.metrics import F1Score

#---------------------------------------------
#이미지 크기 조정 크기
IMG_SIZE = 224
#배치 싸이즈
BATCH_SIZE = 16
#epochs
EPOCHS =  100
EPOCHS_HEAD = 10
patience_head = 3
patience = 10
LAYERS_TRAINABLE = 1 # 트레인 가능한 갯수
backbone =  'efficientnetb4' #'resnet50'
DEFAULT_LABEL_FILE =  "./LPR_Total_Labels.txt"
OBJECT_TYPES = ['vr', 'or'] #['ch','hr', 'vr', 'or']
OBJECT_DETECTION_API_PATH = 'F://SPB_Data//RealTimeObjectDetection-main'
STARTLAYER = 124
MONITOR = 'val_acc'  # val_acc val_loss val_f1

#---------------------------------------------

if MONITOR == 'val_acc':
    op_mode = 'max'
elif MONITOR == 'val_loss':
    op_mode = 'min'
elif MONITOR == 'val_f1':
    op_mode = 'max'
else:
    op_mode = 'max'
    
    
class CustomEarlyStopping(EarlyStopping):
    def update_patience(self, new_patience):
        """
        학습 도중에 patience 값을 업데이트하는 메소드입니다.
        
        Args:
            new_patience (int): 새롭게 적용할 patience 값
        """
        if self.verbose > 0:
            print(f"Updating patience from {self.patience} to {new_patience}.")
        self.patience = new_patience

# ---------------------------------
# imgaug 시퀀스(추가 증강) 정의
# ---------------------------------
aug_seq = iaa.Sequential([
    iaa.AdditiveGaussianNoise(scale=(0, 0.02*255)),  # 가우시안 노이즈
    iaa.Cutout(nb_iterations=1, size=0.05),          # 5% 영역 가리기
    iaa.PerspectiveTransform(scale=(0.01, 0.03))     # 퍼스펙티브(원근) 변환
])

def imgaug_augmentation(img):
    """
    ImageDataGenerator의 preprocessing_function으로 사용할 함수.
    img: 0~1 범위를 갖는 float32 ndarray (Keras 내부 전처리)로 들어옴
    """
    img_uint8 = (img * 255).astype(np.uint8)
    augmented = aug_seq(image=img_uint8)
    return augmented.astype(np.float32) / 255.0

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)

base_dir = os.path.join(ROOT_DIR,'datasets','out1')
if not os.path.isdir(base_dir):
    os.mkdir(base_dir)

logs_dir = os.path.join(ROOT_DIR,'logs')
if not os.path.isdir(logs_dir):
    os.mkdir(logs_dir)

for DEFAULT_OBJ_TYPE in OBJECT_TYPES :
    
    for tainableLen in range(STARTLAYER, STARTLAYER + LAYERS_TRAINABLE):
    
        for nodecnt in [2048, 1024]:
            parser = argparse.ArgumentParser(
                description="object split and save in jpeg and annotation files")
            parser.add_argument("-l",
                                "--labelfile",
                                help="Label file where the text files are stored.",
                                type=str,default=DEFAULT_LABEL_FILE)
            parser.add_argument("-t",
                                "--object_type",
                                help="object type ch : character n: number r: region",
                                type=str,default=DEFAULT_OBJ_TYPE)
            args = parser.parse_args()
            
            fLabels = pd.read_csv(args.labelfile, header=None)
            LABEL_FILE_CLASS = fLabels[0].values.tolist()
            LABEL_FILE_HUMAN_NAMES = fLabels[1].values.tolist()
            CLASS_DIC = dict(zip(LABEL_FILE_CLASS, LABEL_FILE_HUMAN_NAMES))
            
            CH_CLASS = LABEL_FILE_CLASS[LABEL_FILE_CLASS.index('Ga'):LABEL_FILE_CLASS.index('Cml') + 1] + \
                       LABEL_FILE_CLASS[LABEL_FILE_CLASS.index('Gang'):LABEL_FILE_CLASS.index('Heung') + 1] 
            NUM_CLASS = LABEL_FILE_CLASS[LABEL_FILE_CLASS.index('n1'):LABEL_FILE_CLASS.index('n0') + 1]
            REGION_CLASS = LABEL_FILE_CLASS[LABEL_FILE_CLASS.index('vSeoul'):LABEL_FILE_CLASS.index('UlSan6') + 1]
            VREGION_CLASS = LABEL_FILE_CLASS[LABEL_FILE_CLASS.index('vSeoul'):LABEL_FILE_CLASS.index('vDiplomacy') + 1]
            HREGION_CLASS = LABEL_FILE_CLASS[LABEL_FILE_CLASS.index('hSeoul'):LABEL_FILE_CLASS.index('hDiplomacy') + 1]
            OREGION_CLASS = LABEL_FILE_CLASS[LABEL_FILE_CLASS.index('OpSeoul'):LABEL_FILE_CLASS.index('OpUlSan') + 1]
            REGION6_CLASS = LABEL_FILE_CLASS[LABEL_FILE_CLASS.index('Seoul6'):LABEL_FILE_CLASS.index('UlSan6') + 1]
            
            class_str = None
            class_label = []
            model_dir = None
            categorie_filename = None 
            if args.object_type == 'ch':
                class_label = CH_CLASS
                class_str = "character"
                model_dir = 'char_model'
                categorie_filename = 'character_categories.txt'
            elif args.object_type == 'n':
                class_label = NUM_CLASS
                class_str = "number"
                model_dir = 'n_model'
                categorie_filename = 'number_categories.txt'
            elif args.object_type == 'r':
                class_label = REGION_CLASS
                class_str = "region"
                model_dir = 'r_model'
                categorie_filename = 'region_categories.txt'
            elif args.object_type == 'vr':
                class_label = VREGION_CLASS
                class_str = "vregion"
                model_dir = 'vreg_model'
                categorie_filename = 'vregion_categories.txt'
            elif args.object_type == 'hr':
                class_label = HREGION_CLASS
                class_str = "hregion"
                model_dir = 'hreg_model'
                categorie_filename = 'hregion_categories.txt'
            elif args.object_type == 'or':
                class_label = OREGION_CLASS
                class_str = "oregion"
                model_dir = 'oreg_model'
                BATCH_SIZE = 4
                categorie_filename = 'oregion_categories.txt'
            elif args.object_type == 'r6':
                class_str = "region6"
                model_dir = 'reg6_model'
                categorie_filename = 'region6_categories.txt'
                class_label = REGION6_CLASS      
            else:
                print("{0} type is Not supported".format(args.object_type))
                class_str = "Not supported"
                sys.exit(0)
                
            class_dir = os.path.join(base_dir,class_str)
            if not os.path.isdir(class_dir):
                os.mkdir(class_dir)
                
            train_dir = os.path.join(base_dir,class_str,'train')
            if not os.path.isdir(train_dir):
                os.mkdir(train_dir)
            
            validation_dir = os.path.join(base_dir,class_str,'validation')
            if not os.path.isdir(validation_dir):
                os.mkdir(validation_dir)
            
            test_dir = os.path.join(base_dir,class_str,'test')
            if not os.path.isdir(test_dir):
                os.mkdir(test_dir)
            
            categorie_list = os.listdir(train_dir)
            categories = []
            categorie_list = natsort.natsorted(categorie_list)
            for categorie in categorie_list:
                categories.append(categorie)
            
            categories_str = class_str + '_categories.txt'
            with open(categories_str, "w") as f:
                for categorie in categories:
                    f.write(categorie + '\n')
            
            train_data_count =  sum(len(files) for _, _, files in os.walk(train_dir))
            val_data_count = sum(len(files) for _, _, files in os.walk(validation_dir))
            
            model = get_FineTuneModel(backbone, len(categories), tainableLen, nodecnt, dropout_rate=0.33, basemodel_trainable = False)
            
            print("node cnt : {}".format(nodecnt))
            
            try:
                print("...Previous weight data...")
            except:
                print("...New weight data...")
                pass
            
            def get_model_path(model_type, backbone="resnet50"):
                main_path = "trained"
                if not os.path.exists(main_path):
                    os.makedirs(main_path)
                model_path = os.path.join(main_path, 
                    "{}_{}_{}_finetune_{}_node_{}_weights_".format(model_type, backbone,
                                                                   datetime.now().strftime("%Y%m%d-%H%M%S"),
                                                                   tainableLen,nodecnt))
                return model_path
            
            def get_log_path(model_type, backbone="vgg16", custom_postfix=""):
                log_dname = "logs"
                log_path = os.path.join(log_dname, 
                    "{}_{}{}".format(model_type, backbone, custom_postfix),
                    "{}".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
                return log_path
            
            def colortogrey(image):
                image = np.array(image)
                hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                return Image.fromarray(hsv_image)
            
            ### 여기서부터 ImageDataGenerator 수정된 부분 ###
            train_datagen = ImageDataGenerator(
                rotation_range=5,            # -5도 ~ +5도
                width_shift_range=0.1,       # 10% 이내 이동
                height_shift_range=0.1,      # 10% 이내 이동
                shear_range=5,               # -5도 ~ +5도 기울이기
                zoom_range=[0.9, 1.1],       # 0.9 ~ 1.1배 확대/축소
                fill_mode="nearest",
                brightness_range=[0.9, 1.1], # 90% ~ 110% 밝기
                # horizontal_flip=False,
                # preprocessing_function=colortogrey, 
                #preprocessing_function=imgaug_augmentation  ### 추가: imgaug 적용
            )
            
            valid_datagen = ImageDataGenerator()
            
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(IMG_SIZE, IMG_SIZE),
                batch_size=BATCH_SIZE,
                shuffle=True,
                seed=42,
                class_mode='categorical',
                classes=categories
            )
            
            validation_generator = valid_datagen.flow_from_directory(
                validation_dir,
                target_size=(IMG_SIZE, IMG_SIZE),
                batch_size=BATCH_SIZE,
                shuffle=True,
                seed=42,
                class_mode='categorical',
                classes=categories
            )
            
            
            
            
            # 1) ReduceLROnPlateau Callback 정의
            reduce_lr = ReduceLROnPlateau(
                monitor=MONITOR,  # 또는 'val_acc'
                factor=0.1,          # 학습률을 5%로 감소
                patience=5,          # 3 에폭 동안 개선 없으면 학습률 낮춤
                verbose=1,           # 로그 출력 여부
                mode=op_mode,         # auto, min, max
                min_lr=1e-6          # 학습률이 너무 작아지는 상황 방지
            )
            
            f1_metric = F1Score(num_classes=len(categories), average='macro', name='f1')
            
            
            model.summary()

            model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(lr=1e-3),
                          metrics=["acc", f1_metric])
            
            earlystopping = CustomEarlyStopping(monitor=MONITOR, patience=patience_head, mode=op_mode,restore_best_weights=True)
            
            log_path = get_log_path(class_str, backbone)
            model_sub_path_str = get_model_path(class_str, backbone = backbone)
            
            weight_filename =  model_sub_path_str + "epoch_{epoch:03d}_val_acc_{val_acc:.4f}_val_loss_{val_loss:.4f}_val_f1_{val_f1:.4f}.h5"
            checkpoint_callback = ModelCheckpoint(filepath=weight_filename,
                                                  monitor=MONITOR,
                                                  save_freq='epoch',
                                                  save_best_only=True,
                                                  verbose=1,
                                                  mode= op_mode, #'auto',
                                                  save_weights_only=True)
            
            tensorboard_callback = TensorBoard(log_dir=log_path)
            
            class CustomHistory(tf.keras.callbacks.Callback):
                def init(self, logs={}):
                    self.train_loss = []
                    self.val_loss = []
                    self.train_acc = []
                    self.val_acc = []
                    # F1 기록용 리스트 추가
                    self.train_f1   = []
                    self.val_f1     = []
                def on_epoch_end(self, epoch, logs={}):
                    if len(self.val_acc):
                        global weight_filename
                                           
                        if MONITOR == 'val_acc':
                            if logs.get('val_acc') > max(self.val_acc):                             
                                weight_filename = (model_sub_path_str +
                                    "epoch_{0:03d}_val_acc_{1:.4f}_val_loss_{2:.4f}_val_f1_{3:.4f}.h5".format(epoch+1,logs.get('val_acc'),logs.get('val_loss'),logs.get('val_f1', 0)))
                        elif MONITOR == 'val_loss':
                            if logs.get('val_loss') < min(self.val_loss):
                                weight_filename = (model_sub_path_str +
                                    "epoch_{0:03d}_val_acc_{1:.4f}_val_loss_{2:.4f}_val_f1_{3:.4f}.h5".format(epoch+1,logs.get('val_acc'),logs.get('val_loss'),logs.get('val_f1', 0)))
                        elif MONITOR == 'val_f1':
                            # val_f1이 기존보다 커지면 best model 갱신
                            if logs.get('val_f1') and (logs.get('val_f1') > max(self.val_f1) or logs.get('val_acc') > max(self.val_acc)) :
                                weight_filename = (
                                    model_sub_path_str +
                                    "epoch_{0:03d}_val_acc_{1:.4f}_val_loss_{2:.4f}_val_f1_{3:.4f}.h5"
                                    .format(epoch+1,
                                            logs.get('val_acc', 0),
                                            logs.get('val_loss', 0),
                                            logs.get('val_f1', 0))
                                )
                                

                    self.train_loss.append(logs.get('loss'))
                    self.val_loss.append(logs.get('val_loss'))
                    self.train_acc.append(logs.get('acc'))
                    self.val_acc.append(logs.get('val_acc'))
                    # f1 추가
                    #  - fit() 시 metrics=['acc','f1_metric'] 라면 logs.get('f1'), logs.get('val_f1')
                    self.train_f1.append(logs.get('f1'))
                    self.val_f1.append(logs.get('val_f1'))
            
            custom_hist = CustomHistory()
            custom_hist.init()
            """"
            class_weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(train_generator.classes),
                y=train_generator.classes
            )
            class_weights = {i: class_weights[i] for i in range(len(class_weights))}
            """
            steps_per_epoch = int(train_data_count / BATCH_SIZE)
            validation_steps = int(val_data_count / BATCH_SIZE)
            
            history1 = model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=EPOCHS_HEAD,
                validation_data=validation_generator,
                validation_steps=validation_steps,
                # class_weight=class_weights,
                callbacks=[checkpoint_callback, tensorboard_callback, earlystopping, custom_hist,reduce_lr]
            )

            # 실제로 학습이 종료된 마지막 에포크 인덱스
            last_trained_epoch = history1.epoch[-1]  # 예: 6 (0-based)

            print("첫번째 FC의 훈련이 끝났습니다. {} epoch".format(last_trained_epoch))

            # 다음 학습을 이어서 하고 싶다면 +1
            initial_epoch_for_second_fit = last_trained_epoch + 1
            
            acc1 = history1.history['acc']
            val_acc1 = history1.history['val_acc']
            loss1 = history1.history['loss']
            val_loss1 = history1.history['val_loss']
            f1_1 = history1.history['f1']
            val_f1_1 = history1.history['val_f1']

            
            #model = get_FineTuneModel(backbone, len(categories), tainableLen, nodecnt, dropout_rate=0.33, basemodel_trainable = True)
            #weight_path = os.path.join(weight_filename)
            #model.load_weights(weight_path)
            fineTuneModelTrainable(model, backbone, tainableLen, train_from_block='block1')
            
            model.summary()
            model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(lr=1e-4),
                          metrics=["acc", f1_metric])
            earlystopping.update_patience(new_patience=patience)  #새로운 patience값으로 업데이트를 한다.
            reduce_lr2 = ReduceLROnPlateau(
                monitor=MONITOR,  # 또는 'val_acc'
                factor=0.1,          # 학습률을 5%로 감소
                patience=5,          # 3 에폭 동안 개선 없으면 학습률 낮춤
                verbose=1,           # 로그 출력 여부                
                mode=op_mode,         # auto, min, max
                min_lr=1e-7          # 학습률이 너무 작아지는 상황 방지
            )
            checkpoint_callback2 = ModelCheckpoint(filepath=weight_filename,
                                                  monitor=MONITOR,
                                                  save_freq='epoch',
                                                  save_best_only=True,
                                                  verbose=1,
                                                  mode= op_mode, #'auto',
                                                  save_weights_only=True)
            custom_hist2 = CustomHistory()
            custom_hist2.init()
            tensorboard_callback2 = TensorBoard(log_dir=log_path)
            history2 = model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=EPOCHS,
                validation_data=validation_generator,
                validation_steps=validation_steps,
                initial_epoch=initial_epoch_for_second_fit,
                # class_weight=class_weights,
                callbacks=[checkpoint_callback, tensorboard_callback, earlystopping, custom_hist,reduce_lr2]
            )

            # 두 번째 학습 지표
            acc2 = history2.history['acc']
            val_acc2 = history2.history['val_acc']
            loss2 = history2.history['loss']
            val_loss2 = history2.history['val_loss']
            f1_2 = history2.history['f1']
            val_f1_2 = history2.history['val_f1']

            # -------------------------------------------------------------
            # 1) 각 지표별로 그냥 Python list를 붙여서 이어붙이는 방법
            # -------------------------------------------------------------
            acc = acc1 + acc2
            val_acc = val_acc1 + val_acc2
            loss = loss1 + loss2
            val_loss = val_loss1 + val_loss2
            f1 = f1_1 + f1_2
            val_f1 = val_f1_1 + val_f1_2
            
            epochs_range = range(1, len(acc) + 1)
            print('last weight filename is {}'.format(weight_filename))
            
            model_save_filename = "{}_{}_{}_finetune-model_{}_node_{}_epoch_{}_val_acc_{:.4f}_val_loss_{:.4f}_val_f1_{:.4f}.h5".format(
                class_str, backbone, datetime.now().strftime("%Y%m%d-%H%M%S"),
                tainableLen, nodecnt, len(acc), val_acc[-1],val_loss[-1],val_f1[-1]
            )
            model.save(model_save_filename)
        
            # 이하 폴더 정리 및 결과 파일 복사 로직 등 (생략 or 기존 동일)
            # ...
            
            Labelstr = 'Training acc {:03d} {:d}'.format(tainableLen,nodecnt)
            plt.plot(epochs_range, acc, 'bo', label=Labelstr)
            Labelstr = 'Validation acc {:03d} {:d}'.format(tainableLen,nodecnt)
            plt.plot(epochs_range, val_acc, 'b', label=Labelstr)
            Titlestr = 'Training and validation accuracy {:03d} {:d}'.format(tainableLen,nodecnt)
            plt.title(Titlestr)
            plt.legend()
            plt.figure()
            
            Labelstr = 'Training loss {:03d} {:d}'.format(tainableLen,nodecnt)
            plt.plot(epochs_range, loss, 'ro', label=Labelstr)
            Labelstr = 'Validation loss {:03d} {:d}'.format(tainableLen,nodecnt)
            plt.plot(epochs_range, val_loss, 'r', label=Labelstr)       
            Titlestr = 'Training and validation loss {:03d} {:d}'.format(tainableLen,nodecnt)
            plt.title(Titlestr)
            plt.legend()
            plt.figure()

            Labelstr = 'F1 score {:03d} {:d}'.format(tainableLen,nodecnt)
            plt.plot(epochs_range, f1, 'go', label=Labelstr)
            Labelstr = 'Validation F1 score {:03d} {:d}'.format(tainableLen,nodecnt)
            plt.plot(epochs_range, val_f1, 'g', label=Labelstr)       
            Titlestr = 'Training and validation F1 score {:03d} {:d}'.format(tainableLen,nodecnt)
            plt.title(Titlestr)
            plt.legend()
            plt.figure()
            
            plt.show()
