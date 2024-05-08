#from keras import backend as K
#from keras.optimizers import Adadelta
#from keras.callbacks import EarlyStopping, ModelCheckpoint
import os,sys
from tkinter.messagebox import NO
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Activation,Flatten
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
from KSCallBackModel import CustomModelCheckpoint,CustomHistory
from sklearn.model_selection import StratifiedKFold
K.set_learning_phase(0)

#---------------------------------------------
#이미지 크기 조정 크기
IMG_SIZE = 224
#배치 싸이즈
BATCH_SIZE = 8
#epochs
EPOCHS =  35
patience = 6
LAYERS_TRAINABLE = 10 # 트레인 가능한 갯수
backbone = 'resnet50'
DEFAULT_LABEL_FILE =  "./LPR_Total_Labels.txt" #"./LPR_Labels1.txt"  #라벨 파일이름
OBJECT_TYPES = ['hr', 'vr'] #['ch','hr', 'vr', 'or']
OBJECT_DETECTION_API_PATH = 'C://SPB_Data//RealTimeObjectDetection-main'
STARTLAYER = 120
K_FOLD = 5
FOLD_EPOCHS = 2
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

for DEFAULT_OBJ_TYPE in OBJECT_TYPES :

    for tainableLen in range(STARTLAYER,STARTLAYER + LAYERS_TRAINABLE):
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
        # CH_CLASS = LABEL_FILE_CLASS[21:111]  #문자열 클래스
        # NUM_CLASS = LABEL_FILE_CLASS[11:21]  #숫자 클래스
        # REGION_CLASS = LABEL_FILE_CLASS[111:-1] #지역문자 클래스
        # VREGION_CLASS = LABEL_FILE_CLASS[111:128] #Vertical 지역문자 클래스
        # HREGION_CLASS = LABEL_FILE_CLASS[128:145] #Horizontal 지역문자 클래스
        # OREGION_CLASS = LABEL_FILE_CLASS[145:162] #Orange 지역문자 클래스
        # REGION6_CLASS = LABEL_FILE_CLASS[162:-1] #6 지역문자 클래스
        
        CH_CLASS =  LABEL_FILE_CLASS[LABEL_FILE_CLASS.index('Ga'):LABEL_FILE_CLASS.index('Cml') + 1] + LABEL_FILE_CLASS[LABEL_FILE_CLASS.index('Gang'):LABEL_FILE_CLASS.index('Heung') + 1] #문자열 클래스
        NUM_CLASS = LABEL_FILE_CLASS[LABEL_FILE_CLASS.index('n1'):LABEL_FILE_CLASS.index('n0') + 1]  #숫자 클래스
        REGION_CLASS = LABEL_FILE_CLASS[LABEL_FILE_CLASS.index('vSeoul'):LABEL_FILE_CLASS.index('UlSan6') + 1] #지역문자 클래스
        #VREGION_CLASS = LABEL_FILE_CLASS[LABEL_FILE_CLASS.index('vSeoul'):LABEL_FILE_CLASS.index('vUlSan') + 1] #Vertical 지역문자 클래스
        #HREGION_CLASS = LABEL_FILE_CLASS[LABEL_FILE_CLASS.index('hSeoul'):LABEL_FILE_CLASS.index('hUlSan') + 1] #Horizontal 지역문자 클래스
        VREGION_CLASS = LABEL_FILE_CLASS[LABEL_FILE_CLASS.index('vSeoul'):LABEL_FILE_CLASS.index('vDiplomacy') + 1] #Vertical 지역문자 클래스
        HREGION_CLASS = LABEL_FILE_CLASS[LABEL_FILE_CLASS.index('hSeoul'):LABEL_FILE_CLASS.index('hDiplomacy') + 1] #Horizontal 지역문자 클래스        
        OREGION_CLASS = LABEL_FILE_CLASS[LABEL_FILE_CLASS.index('OpSeoul'):LABEL_FILE_CLASS.index('OpUlSan') + 1] #Orange 지역문자 클래스
        REGION6_CLASS = LABEL_FILE_CLASS[LABEL_FILE_CLASS.index('Seoul6'):LABEL_FILE_CLASS.index('UlSan6') + 1] #6 지역문자 클래스
        
        
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
            #print("{0} type is Not supporeted yet".format(args.object_type))
            #sys.exit(0)
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
            BATCH_SIZE = 4 # 갯수가 작아서 에러가 날수 있으므로...
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
        
        model = get_FineTuneModel(backbone,len(categories),tainableLen)
        
        
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
            model_path = os.path.join(main_path, "{}_{}_{}_finetune_{}_weights_".format(model_type, backbone,datetime.now().strftime("%Y%m%d-%H%M%S"),tainableLen))
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
        
        kfold = StratifiedKFold(n_splits=K_FOLD, shuffle=True)
    
        labels = []
        image_paths = []

        dfilecnt = np.zeros(len(categorie_list))
    
        for index, categorie in enumerate(categorie_list):
            for filename in os.listdir(os.path.join(train_dir,categorie)):
                #labels.append(categorie)
                #image_paths.append(os.path.join(train_dir,categorie,filename))
                dfilecnt[index] =dfilecnt[index] + 1
                
        for index, value in enumerate(dfilecnt):
            if value < K_FOLD:
                print('{}의 파일갯수는{} 입니다'.format(categorie_list[index],value))
                #1개를 validation을 위하여 더 추가로 만든다.
                fn = os.listdir(os.path.join(train_dir,categorie_list[index]))
                if fn:
                    src_size = len(fn)
                    s_idx = 0
                    for i in range(K_FOLD-1):       
                        filename, ext = os.path.splitext(fn[s_idx])
                        srcfn = os.path.join(train_dir,categorie_list[index],fn[s_idx])
                        dstfn = os.path.join(train_dir,categorie_list[index],filename + '_{}'.format(i+1) + ext)
                        shutil.copy(srcfn,dstfn)
                        s_idx = (s_idx + 1) % src_size
                else:
                    print('에러!!! {}의 파일이 없습니다.'.format(categorie_list[index]))
                    
        for index, categorie in enumerate(categorie_list):
            for filename in os.listdir(os.path.join(train_dir,categorie)):
                labels.append(categorie)
                image_paths.append(os.path.join(train_dir,categorie,filename))
            
        df = pd.DataFrame({'image_path':image_paths,'label':labels})

        fold_no = 0
        custom_hist = CustomHistory()
        custom_hist.init(fold_no)
        
        model.summary()
        
        model.compile( loss='categorical_crossentropy', optimizer=Adam(lr=0.001),metrics=["acc"])
        
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=patience)
        
        
        
        # Load weights
        log_path = get_log_path(class_str, backbone)
        model_sub_path_str = get_model_path(class_str)
        
        # weight file format을 지정한다.
        weight_filename =  model_sub_path_str + "epoch_{epoch:03d}_val_acc_{val_acc:.4f}.h5"
        checkpoint_callback = CustomModelCheckpoint(filepath=weight_filename , monitor="val_acc",save_best_only=True, mode='max', patience=patience)
        
        is_earlystopping = False

        for _ in  range(int(EPOCHS/(K_FOLD*FOLD_EPOCHS))):
            for fold, (train_idx, valid_idx) in enumerate(kfold.split(df, df['label'])):

                train_df = df.iloc[train_idx]
                valid_df = df.iloc[valid_idx]

                train_datagen = ImageDataGenerator(
                                        rotation_range=45,
                                         width_shift_range=0.2,
                                         height_shift_range=0.2,
                                         shear_range=0.2,
                                         zoom_range=[0.7,1.0],
                                         #horizontal_flip=True,
                                         #preprocessing_function = colortogrey,
                                         brightness_range=[0.2,1.0])
            
                valid_datagen = ImageDataGenerator()
                
                train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                    x_col='image_path',
                                                    y_col='label',
                                                    target_size=(IMG_SIZE,IMG_SIZE),
                                                    class_mode='categorical'
                                                    )
                
                
                
                validation_generator = valid_datagen.flow_from_dataframe(dataframe=valid_df,
                                                    x_col='image_path',
                                                    y_col='label',
                                                    target_size=(IMG_SIZE,IMG_SIZE),
                                                    class_mode='categorical')
            
                    
                tensorboard_callback = TensorBoard(log_dir=log_path)
                
                #calculate steps_per_epoch and validation_steps
                steps_per_epoch = int(train_generator.n/train_generator.batch_size)
                validation_steps = int(validation_generator.n/validation_generator.batch_size)
                
                #weight 파일이 있으면 읽어서 갱신한다.
                
                last_weight_fn = checkpoint_callback.get_weight_filename()
                if last_weight_fn:
                    if os.path.isfile(last_weight_fn):
                        print("가장 최근 weight 파일 읽기: {}".format(last_weight_fn))
                        model.load_weights(last_weight_fn)
                print('K-Fold:{}'.format(fold+1))   
                history = model.fit(train_generator,
                                            steps_per_epoch=steps_per_epoch,
                                            epochs=FOLD_EPOCHS,
                                            validation_data=validation_generator,
                                            validation_steps=validation_steps,
                                            callbacks=[checkpoint_callback,tensorboard_callback,earlystopping,custom_hist])
                acc = custom_hist.train_acc
                val_acc = custom_hist.val_acc
                loss = custom_hist.train_loss
                val_loss = custom_hist.val_loss
                
                epochs = range(1, len(custom_hist.train_acc) + 1)
                
                # Increase fold number
                fold_no += 1
                
                #earlystopping 조건이면 루프를 빠져 나간다.
                is_earlystopping  = checkpoint_callback.earlystopping()
                if is_earlystopping:
                    break
                
            if is_earlystopping:
                break

        
        model_save_filename = "{}_{}_{}_finetune-model_{}_epoch_{}_val_acc_{:.4f}.h5".format(class_str,backbone,datetime.now().strftime("%Y%m%d-%H%M%S"),tainableLen,len(acc),val_acc[-1])
        model.save(model_save_filename,)
                

        #기존 폴더 아래 있는 출력 폴더를 지운다.
        model_path = os.path.join(OBJECT_DETECTION_API_PATH, model_dir)
        
        if not os.path.isdir(model_path) :
            createFolder(model_path)
            
        model_list = os.listdir(model_path)
        if( len(model_list)):
            for fn in model_list:
                os.remove(os.path.join(model_path,fn))
                
        Labelstr = 'Training acc {:03d}'.format(tainableLen)
        plt.plot(epochs, acc, 'bo', label = Labelstr)
        Labelstr = 'Validation acc {:03d}'.format(tainableLen)
        plt.plot(epochs, val_acc, 'b', label = Labelstr)

        Titlestr = 'K-fold: {} Training and validation accuracy {:03d}'.format(fold_no,tainableLen)
        plt.title(Titlestr)
        plt.legend()
        plt.figure()
        Labelstr = 'Training loss {:03d}'.format(tainableLen)
        plt.plot(epochs, loss, 'bo', label = Labelstr)
        Labelstr = 'Validation loss {:03d}'.format(tainableLen)
        plt.plot(epochs, val_loss, 'b', label = Labelstr)
        
        Titlestr = 'K-fold: {} Training and validation loss {:03d}'.format(fold_no, tainableLen)
        plt.title(Titlestr)
        plt.legend()
        plt.figure()
        
        plt.show()