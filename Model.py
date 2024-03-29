import tensorflow as tf
from keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.layers import Reshape, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from parameter import *
K.set_learning_phase(0)



def get_Model(categories_len, trainablelayerLen = 95):
    input_shape = (img_w, img_h, 3)     # (128, 64, 1)
    
    # ResNet50 불러오기 -> include_top = False로 바꾸는 것이 포인트
   
    base_model = ResNet50(include_top=False, pooling = 'avg' , input_shape = input_shape, weights = 'imagenet')
    base_model.trainable = True
    
    endLayer = len(base_model.layers)

    for layer in base_model.layers[: - trainablelayerLen]:    
          layer.trainable = False 
    
    base_model.summary()
    model = tf.keras.models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(2048,activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(categories_len,activation='softmax'))

    return model

def get_FineTuneModel(categories_len, trainableLen):
    input_shape = (img_w, img_h, 3)     # (128, 64, 1)
    
    # ResNet50 불러오기 -> include_top = False로 바꾸는 것이 포인트
   
    base_model = ResNet50(include_top=False, pooling = 'avg' , input_shape = input_shape, weights = 'imagenet')
    base_model.trainable = True
    
    endLayer = len(base_model.layers)

    for layer in base_model.layers[: - trainableLen]:    
         layer.trainable = False
         #print('Layer name {} untrainable'.format(layer.name)) 
    
    base_model.summary()
    model = tf.keras.models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(2048,activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(categories_len,activation='softmax'))

    return model



