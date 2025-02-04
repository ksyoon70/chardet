import tensorflow as tf
from keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB4  # 추가
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.layers import Reshape, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from parameter import *
K.set_learning_phase(0)



def get_Model(backbone, categories_len, trainablelayerLen = 95):
    input_shape = (img_w, img_h, 3)     # (128, 64, 1)
    
    # ResNet50 불러오기 -> include_top = False로 바꾸는 것이 포인트
   
    if(backbone=='resnet50'):
        base_model = ResNet50(include_top=False, pooling = 'avg' , input_shape = input_shape, weights = 'imagenet')
    else:
        base_model = ResNet101(include_top=False, pooling = 'avg' , input_shape = input_shape, weights = 'imagenet')
        
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

def get_FineTuneModel(backbone, categories_len, trainableLen, nodecnt, dropout_rate = 0.3, train_from_block = 'block5', basemodel_trainable = False):
    input_shape = (img_w, img_h, 3)     # (128, 64, 1)
    
    # ResNet50 불러오기 -> include_top = False로 바꾸는 것이 포인트
   
    if(backbone=='resnet50'):
        base_model = ResNet50(include_top=False, pooling = 'avg' , input_shape = input_shape, weights = 'imagenet')
    elif backbone=='resnet101':
        base_model = ResNet101(include_top=False, pooling = 'avg' , input_shape = input_shape, weights = 'imagenet')
    elif backbone=='efficientnetb0':
        base_model = EfficientNetB0(include_top=False, pooling='avg', 
                                    input_shape=input_shape, weights='imagenet')
    elif backbone=='efficientnetb4':
        base_model = EfficientNetB4(include_top=False, pooling='avg', 
                                    input_shape=input_shape, weights='imagenet')                                   
    else:
        raise ValueError("지원하지 않는 backbone 입니다: {}".format(backbone))
    
    if basemodel_trainable == True:

        base_model.trainable = True
        
        endLayer = len(base_model.layers)
        
        # -------------------------------------------
        # 1) 특정 블록부터 trainable = True
        #    그 이전 레이어는 trainable = False
        # -------------------------------------------
        found_block = False
        if backbone=='efficientnetb0' or backbone=='efficientnetb4': 
            found_block = False
            for layer in base_model.layers:
            # 예) 'block5'라는 문자열로 시작하는 레이어 이름을 만나면 그 다음부터 True
                if layer.name.startswith(train_from_block):
                    found_block = True
                layer.trainable = found_block

                #if isinstance(layer, tf.keras.layers.BatchNormalization):
                    #layer.trainable = False
        else:
            for layer in base_model.layers[: - trainableLen]:
                layer.trainable = False
            for layer in base_model.layers[trainableLen:]:
                layer.trainable = True
                #if isinstance(layer, tf.keras.layers.BatchNormalization):
                    #layer.trainable = False
    else:
        base_model.trainable = False

    print("=== {} Layer 목록 ===".format(backbone))
    for i, layer in enumerate(base_model.layers):
            print(f"{i:3d} | {layer.name} | trainable={layer.trainable}")
  
    #base_model.summary()
    model = tf.keras.models.Sequential()
    model.add(base_model)
    # 활성화 함수 설정
    activation_fn = 'swish' if (backbone == 'efficientnetb0' or backbone == 'efficientnetb4') else 'relu'
    model.add(layers.Flatten())
    # Dense 레이어 구조
    model.add(layers.Dense(nodecnt, activation=None))
    #model.add(layers.Dense(nodecnt))
    model.add(layers.BatchNormalization())
    #model.add(layers.ReLU())
    model.add(layers.Activation(activation_fn))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(nodecnt // 2, activation=None))
    #model.add(layers.Dense(nodecnt // 2))
    model.add(layers.BatchNormalization())
    #model.add(layers.ReLU())
    model.add(layers.Activation(activation_fn))
    model.add(layers.Dropout(dropout_rate))

    # 최종 출력 레이어
    model.add(layers.Dense(categories_len, activation='softmax'))

    return model

def fineTuneModelTrainable(model,backbone,trainableLen,train_from_block = 'block5'):
    # (1) base_model(EfficientNetB0)은 model.layers[0] 에 해당함
    base_model = model.layers[0]  # EfficientNetB0 (include_top=False)
    fc_layers  = model.layers[1:] # FC 부분(Dense, BN, Dropout 등)

    base_model.trainable = True
    if backbone=='efficientnetb0' or backbone=='efficientnetb4': 
        found_block = False
        for layer in base_model.layers:
        # 예) 'block5'라는 문자열로 시작하는 레이어 이름을 만나면 그 다음부터 True
            if layer.name.startswith(train_from_block):
                found_block = True
            layer.trainable = found_block

            #if isinstance(layer, tf.keras.layers.BatchNormalization):
                #layer.trainable = False
    else:        
        for layer in base_model.layers[: - trainableLen]:
            layer.trainable = False
        #for layer in base_model.layers[trainableLen:]:
            #if isinstance(layer, tf.keras.layers.BatchNormalization):
                #layer.trainable = False

    #base_model.summary()
    print("=== {} Layer 목록 ===".format(backbone))
    for i, layer in enumerate(base_model.layers):
            print(f"{i:3d} | {layer.name} | trainable={layer.trainable}")

    # 3) FC 부분은 언제나 학습해야 하므로 trainable = True
    for layer in fc_layers:
        layer.trainable = True
        #if isinstance(layer, tf.keras.layers.BatchNormalization):
            #layer.trainable = False



