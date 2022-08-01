import tensorflow as tf
from keras import backend as K
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.layers import Reshape, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from parameter import *
K.set_learning_phase(0)



def get_Model(training):
    input_shape = (img_w, img_h, 3)     # (128, 64, 1)

    # Make Networkw
    inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)
    
    # ResNet50 불러오기 -> include_top = False로 바꾸는 것이 포인트
    base_model = ResNet50(include_top=False, pooling = 'avg' , input_shape = input_shape, weights = 'imagenet')
    C4 =  base_model.get_layer('conv4_block6_out').output
    base_model.trainable = False
    
    rest50_model = Model(inputs=base_model.input,outputs= C4)
    
    rest50_model.summary()
    
    
    for layer in rest50_model.layers:
        layer.trainable = False

    rest50_model.summary()
    
    inner = tf.keras.applications.resnet50.preprocess_input(inputs)
    
    inner = rest50_model.output
    

    #base_model(inputs, training = False)

    inner = Conv2D(2048, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(inner)  # (None, 32, 4, 512)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)


    # transforms RNN output to character activations:
    inner = Dense(num_classes, kernel_initializer='he_normal',name='dense2')(inner) #(None, 32, 63)
    y_pred = Activation('softmax', name='softmax')(inner)

    return Model(inputs=[rest50_model.inputs], outputs=y_pred)



