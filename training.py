#from keras import backend as K
#from keras.optimizers import Adadelta
#from keras.callbacks import EarlyStopping, ModelCheckpoint
import os,sys
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
K.set_learning_phase(0)

#---------------------------------------------
#이미지 크기 조정 크기
IMG_SIZE = 224
#배치 싸이즈
BATCH_SIZE = 20
#epochs
EPOCHS =  400
backbone = 'resnet50'
DEFAULT_LABEL_FILE = "./LPR_Labels1.txt"  #라벨 파일이름
#---------------------------------------------

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)


#디렉토리를 만듭니다.
base_dir = os.path.join(ROOT_DIR,'datasets')
if not os.path.isdir(base_dir):
    os.mkdir(base_dir)

train_dir = os.path.join(base_dir,'train')
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)

validation_dir = os.path.join(base_dir,'validation')
if not os.path.isdir(validation_dir):
    os.mkdir(validation_dir)

test_dir = os.path.join(base_dir,'test')
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)


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

#categorie_list = check_class
categorie_list = os.listdir(train_dir)
categories = []
categorie_list = natsort.natsorted(categorie_list)
for categorie in categorie_list:
    categories.append(categorie)
    

train_data_count =  sum(len(files) for _, _, files in os.walk(train_dir))
val_data_count = sum(len(files) for _, _, files in os.walk(validation_dir))
# # Model description and training

#model = get_Model(training=True)
base_model = tf.keras.applications.resnet50.ResNet50(include_top=False, pooling = 'avg' , input_shape = (IMG_SIZE,IMG_SIZE ,3), weights = 'imagenet')
base_model.trainable = False

for layer in base_model.layers:
    layer.trainable = False 

"""
inputs = Input(shape=(IMG_SIZE,IMG_SIZE,3))
x = tf.keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE)(inputs)
x = tf.keras.applications.resnet50.preprocess_input(inputs)
x = base_model(x, training = False)
x = Flatten()(x)
# Fully Connected에 온전하게 학습을 위해 펼쳐준다	
tf.keras.layers.Dense(512,activation='relu')(x)
x = tf.keras.layers.BatchNormalization() (x)
x = tf.keras.layers.Dropout(rate=0.5)(x)
outputs = Dense(len(categories), activation = 'softmax')(x)         # Softmax 함수로 10개 분류하는 분류기 
model = tf.keras.Model(inputs, outputs)               # model_res 란 이름의 인풋과 아웃풋이 정해진 모델
"""


model = tf.keras.models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(2048,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(categories),activation='softmax'))


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
    model_path = os.path.join(main_path, "{}_{}_model_weights.h5".format(model_type, backbone))
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
    return "logs/{}_{}{}/{}".format(model_type, backbone, custom_postfix, datetime.now().strftime("%Y%m%d-%H%M%S"))

train_datagen = ImageDataGenerator(
                            rescale=1.0,
                            rotation_range=5,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2)

#train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1.0)


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

model.compile( loss='categorical_crossentropy', optimizer='adam',metrics=["acc"])

early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# Load weights
log_path = get_log_path("cls", backbone)
model_path = get_model_path("cls")
checkpoint_callback = ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, save_weights_only=True)
tensorboard_callback = TensorBoard(log_dir=log_path)

class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(train_generator.classes), 
                train_generator.classes) 

class_weights = {i : class_weights[i] for i in range(len(categories))}
#calculate steps_per_epoch and validation_steps
steps_per_epoch = int(train_data_count/BATCH_SIZE)
validation_steps = int(val_data_count/BATCH_SIZE)
history = model.fit(train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=EPOCHS,
                              validation_data=validation_generator,
                              validation_steps=validation_steps,
                              #class_weight=class_weights,
                              callbacks=[checkpoint_callback, tensorboard_callback])


model.save('chardet_model.h5',)



acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)


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