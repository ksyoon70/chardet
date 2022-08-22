#from keras import backend as K
#from keras.optimizers import Adadelta
#from keras.callbacks import EarlyStopping, ModelCheckpoint
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
from tensorflow.keras.layers import Input, Dense, Activation,Flatten
from tensorflow.keras.layers import Reshape, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from Model import get_Model
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

K.set_learning_phase(0)

#---------------------------------------------
#이미지 크기 조정 크기
IMG_SIZE = 224
#배치 싸이즈
BATCH_SIZE = 32
#epochs
EPOCHS =  50
backbone = 'resnet50'
DEFAULT_LABEL_FILE = "./LPR_Labels1.txt"  #라벨 파일이름
DEFAULT_OBJ_TYPE = 'ch'#'ch'
OBJECT_DETECTION_API_PATH = 'C://SPB_Data//RealTimeObjectDetection-main'
#---------------------------------------------

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)


#디렉토리를 만듭니다.
base_dir = os.path.join(ROOT_DIR,'datasets')
if not os.path.isdir(base_dir):
    os.mkdir(base_dir)

logs_dir = os.path.join(ROOT_DIR,'logs')
#로그 디렉토리 만들기
if not os.path.isdir(logs_dir):
	os.mkdir(logs_dir)

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="object split and save in jpeg and annotation files")
parser.add_argument("-l",
                    "--labelfile",
                    help="Label file where the text files are stored.",
                    type=str,default=DEFAULT_LABEL_FILE)
# 검색할 object type를 설정한다. 
parser.add_argument("-t",
                    "--object_type",
                    help="object type ch : character n: number r: region", type=str,default=DEFAULT_OBJ_TYPE)


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


class_str = None   #클래스의 이름을 저장한다.
class_label = [];
model_dir = None
categorie_filename = None 
if args.object_type == 'ch':        #문자 검사
    class_label = CH_CLASS
    class_str = "character"
    model_dir = 'char_model'
    categorie_filename = 'character_categories.txt'
elif args.object_type == 'n':       #숫자검사
    class_label = NUM_CLASS
    class_str = "number"
    model_dir = 'n_model'
    categorie_filename = 'number_categories.txt'
    print("{0} type is Not supporeted yet".format(args.object_type))
    sys.exit(0)
elif args.object_type == 'r':       #지역문자 검사
    class_label = REGION_CLASS
    class_str = "region"
    model_dir = 'r_model'
    categorie_filename = 'region_categories.txt'
elif args.object_type == 'vr':       #v 지역문자 검사
    class_label = VREGION_CLASS
    class_str = "vregion"
    model_dir = 'vreg_model'
    categorie_filename = 'vregion_categories.txt'
elif args.object_type == 'hr':       #h 지역문자 검사
    class_label = HREGION_CLASS
    class_str = "hregion"
    model_dir = 'hreg_model'
    categorie_filename = 'hregion_categories.txt'
elif args.object_type == 'or':       #o 지역문자 검사
    class_label = OREGION_CLASS
    class_str = "oregion"
    model_dir = 'oreg_model'
    categorie_filename = 'oregion_categories.txt'
elif args.object_type == 'r6':       #6 지역문자 검사
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
    for categorie in categories :
        f.write(categorie + '\n')
   

train_data_count =  sum(len(files) for _, _, files in os.walk(train_dir))
val_data_count = sum(len(files) for _, _, files in os.walk(validation_dir))
# # Model description and training

model = get_Model(len(categories))


try:
   # model.load_weights('LSTM+BN4--26--0.011.hdf5')
    print("...Previous weight data...")
except:
    print("...New weight data...")
    pass

def get_model_path(model_type, backbone="resnet50"):
    """Generating model path from model_type value for save/load model weights.
    inputs:
        model_type = "rpn", "faster_rcnn"
        backbone = "vgg16", "mobilenet_v2"
    outputs:
        model_path = os model path, for example: "trained/rpn_vgg16_model_weights.h5"
    """
    main_path = "trained"
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    model_path = os.path.join(main_path, "{}_{}_{}_weights_".format(model_type, backbone,datetime.now().strftime("%Y%m%d-%H%M%S")))
    return model_path

def get_log_path(model_type, backbone="vgg16", custom_postfix=""):
    """Generating log path from model_type value for tensorboard.
    inputs:
        model_type = "rpn", "faster_rcnn"
        backbone = "vgg16", "mobilenet_v2"
        custom_postfix = any custom string for log folder name
    outputs:
        log_path = tensorboard log path, for example: "logs/rpn_mobilenet_v2/{date}"
    """
    log_dname = "logs"
    log_path = os.path.join(log_dname,"{}_{}{}".format(model_type, backbone, custom_postfix),"{}".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
    
    return log_path

def colortogray(image):
    image = np.array(image)
    hsv_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    return Image.fromarray(hsv_image)

train_datagen = ImageDataGenerator(
                            rotation_range=30,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=[0.7,1.0],
                             #horizontal_flip=True,
                             #preprocessing_function = colortogrey,
                             brightness_range=[0.2,1.0])

valid_datagen = ImageDataGenerator()


train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(IMG_SIZE,IMG_SIZE),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    seed=42,
                                                    class_mode='categorical',
                                                    classes = categories)


validation_generator = valid_datagen.flow_from_directory(validation_dir,
                                                    target_size=(IMG_SIZE,IMG_SIZE),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    seed=42,
                                                    class_mode='categorical',
                                                    classes = categories)

model.summary()

model.compile( loss='categorical_crossentropy', optimizer=Adam(lr=0.001),metrics=["acc"])

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10)
# Load weights
log_path = get_log_path(class_str, backbone)
model_sub_path_str = get_model_path(class_str)


weight_filename = model_sub_path_str + "epoch_{epoch:03d}_val_acc_{val_acc:.3f}.h5"
checkpoint_callback = ModelCheckpoint(filepath=weight_filename , monitor="val_acc", save_freq='epoch',save_best_only=True, verbose=1, mode='auto' ,save_weights_only=True)

tensorboard_callback = TensorBoard(log_dir=log_path)

class CustomHistory(tf.keras.callbacks.Callback):
    def init(self, logs={}):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        
    def on_epoch_end(self, epoch, logs={}):
        if len(self.val_acc):
            if logs.get('val_acc') > max(self.val_acc) :
                global weight_filename
                weight_filename = model_sub_path_str + "epoch_{0:03d}_val_acc_{1:.3f}.h5".format(epoch+1,logs.get('val_acc'))
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        #print('\nepoch={}, 현재 최대 val_acc={}'.format(epoch,max(self.val_acc)))


custom_hist = CustomHistory()
custom_hist.init()


class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(train_generator.classes), 
                train_generator.classes) 

class_weights = {i : class_weights[i] for i in range(len(class_weights))}


#calculate steps_per_epoch and validation_steps
steps_per_epoch = int(train_data_count/BATCH_SIZE)
validation_steps = int(val_data_count/BATCH_SIZE)
history = model.fit(train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=EPOCHS,
                              validation_data=validation_generator,
                              validation_steps=validation_steps,
                              #class_weight=class_weights,
                              callbacks=[checkpoint_callback,tensorboard_callback,earlystopping,custom_hist])


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

print('last weight filename is {}'.format(weight_filename))

model_save_filename = "{}_{}_{}_model_epoch_{}_val_acc_{:.4f}.h5".format(class_str,backbone,datetime.now().strftime("%Y%m%d-%H%M%S"),len(acc),val_acc[-1])
model.save(model_save_filename,)



#기존 폴더 아래 있는 출력 폴더를 지운다.
model_path = os.path.join(OBJECT_DETECTION_API_PATH,model_dir)
model_list = os.listdir(model_path)
if( len(model_list)):
    for fn in model_list:
        os.remove(os.path.join(model_path,fn))
        
#결과 파일을 복사한다.
#weight file 복사
src_fn = weight_filename
dst_fn = os.path.join(model_path,os.path.basename(src_fn))
shutil.copy(src_fn,dst_fn)
# 카테고리 파일 복사
src_fn = categorie_filename
dst_fn = os.path.join(model_path,src_fn)
shutil.copy(src_fn,dst_fn)
# 모델 파일 복사
src_fn = model_save_filename
dst_fn = os.path.join(model_path,src_fn)
shutil.copy(src_fn,dst_fn)


plt.plot(epochs, acc, 'bo', label ='Training acc')
plt.plot(epochs, val_acc, 'b', label ='Validation acc')

plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label ='Training loss')
plt.plot(epochs, val_loss, 'b', label ='Validation loss')

plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.show()