# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:29:59 2024

@author: headway
"""
import os,sys
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', save_best_only=True, mode='max',patience=10, **kwargs):
        super().__init__(filepath, monitor=monitor, save_best_only=save_best_only, **kwargs)
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.epoch = 0
        self.weight_filename = None
        self.patience = patience
        self.update = 0

    def on_epoch_end(self, epoch, logs=None):
        # 원하는 동작을 구현 (예: 특정 조건에서만 저장)
        #if epoch % 5 == 0:
        #    super().on_epoch_end(epoch, logs)
        current_value = logs.get(self.monitor)
        if current_value is None:
            return

        if (self.mode == 'max' and current_value > self.best_value) or (self.mode == 'min' and current_value < self.best_value):
            self.best_value = current_value            
            super().on_epoch_end(self.epoch, logs)
            self.weight_filename = self.filepath.format(epoch=self.epoch + 1, **logs)
            if self.update < self.patience:
                self.update = 0
        else:
            self.update += 1
            
        self.epoch += 1     #자체적으로 epoch 증가
    #마지막 저장한 weight file name을 리턴한다.        
    def get_weight_filename(self):
        return self.weight_filename
    def earlystopping(self):
        return bool(True if self.update >= self.patience else False )
    

class CustomHistory(tf.keras.callbacks.Callback):
    def init(self, fold, logs={}):
        super().__init__()
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.epoch = 0
        
    def on_epoch_end(self, epoch, logs={}):
        """
        if len(self.val_acc):
            if logs.get('val_acc') > max(self.val_acc) :
                global weight_filename
                global fold_no
                weight_filename =  model_sub_path_str + "fold_{0:02d}_epoch_{1:03d}_val_acc_{2:.4f}.h5".format(fold_no, epoch+1,logs.get('val_acc'))
        """
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.epoch += 1
        #print('\nepoch={}, 현재 최대 val_acc={}'.format(epoch,max(self.val_acc)))