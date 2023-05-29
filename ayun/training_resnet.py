import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D
from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout, BatchNormalization
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet import ResNet50, ResNet152, ResNet101
import datetime
import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"]="0"

idx2label = {0: '무증상', 1: '결막염', 2: '궤양성각막질환', 3: '백내장', 4: '비궤양성각막질환',
               5: '색소침착성각막염', 6: '안검내반증', 7: '안검염', 8: '안검종양', 9: '유루증', 10: '핵경화'}

# - rescale: 이미지의 픽셀값 범위를 0에서 1 사이의 값으로 정규화합니다.
# - shear_range: 전단변환(shear transformation)을 적용합니다. 전단변환은 이미지를 수평 또는 수직 방향으로 이동시키는 변환이며, 이동 시에 생기는 빈 공간을 채우기 위해 이미지를 변형시킵니다.
# - rotation_range: 이미지를 임의의 각도로 회전시킵니다. 각도 범위는 0에서 180도까지입니다.
# - width_shift_range와 height_shift_range: 이미지를 수평 또는 수직 방향으로 이동시킵니다. 이동 거리는 전체 이미지 크기에 대한 비율로 지정합니다.
# - zoom_range: 이미지를 임의의 확대/축소 비율로 변환합니다.
# horizontal_flip과 vertical_flip: 이미지를 수평 또는 수직 방향으로 뒤집습니다.
# - validation_split: 검증 데이터셋의 비율을 지정합니다.
# - horizontal_flip : 가로로 뒤집기
# - vertical_flip : 세로로 뒤집기

# /alpaco/vscode/testenv/Scripts/activate.bat


train_path = 'E:/eye_multiclass_notm/dataset/train'
val_path = 'E:/eye_multiclass_notm/dataset/val'
test_path = 'E:/eye_multiclass_notm/test_balanced'

with open('E:/eye_multiclass_notm/dataset/mean_img.pickle', 'rb') as f:
    mean_img = pickle.load(f)
sub_mean_img = lambda image: image - mean_img

batch_size = 32

# 이미지 데이터 제너레이터 생성
train_datagen = ImageDataGenerator(
    # rescale= 1.0/255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    preprocessing_function= sub_mean_img)
datagen = ImageDataGenerator(
    # rescale= 1.0/255,
    preprocessing_function=sub_mean_img)

# 이미지 데이터를 불러오기 위한 제너레이터 생성
train_generator = train_datagen.flow_from_directory(
    directory= train_path,
    target_size=(224, 224),
    batch_size= batch_size,
    shuffle = True,
    class_mode='categorical')
val_generator = datagen.flow_from_directory(
    directory= val_path ,
    target_size=(224, 224),
    batch_size= batch_size,
    shuffle = False,
    class_mode='categorical')
test_generator = datagen.flow_from_directory(
    directory= test_path ,
    target_size=(224, 224),
    batch_size= batch_size,
    shuffle = False,
    class_mode='categorical')

# 모델 학습
base_model50 = ResNet50(include_top=False, input_shape = (224, 224 ,3), weights = 'imagenet')  

base_model50.trainable = True

# model.layers = input -> output 방향으로 리스트 출력
for layer in base_model50.layers[:-20]:
    layer.trainable = False			

inputs = tf.keras.Input(shape=(224, 224, 3))

x = base_model50(inputs, training=False) # batchnorm 부분 update 방지

x = tf.keras.layers.Flatten(input_shape=base_model50.output_shape[1:])(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x= tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model50 = tf.keras.Model(inputs, outputs)
model50.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), loss = 'categorical_crossentropy',
                metrics = ['accuracy', tf.keras.metrics.Recall()]
                )

es = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience = 15, restore_best_weights= True)
model_name = 'res50'
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M") 


log_dir = "D:/alpaco/image_classification/logs/fit/" + current_time + model_name
board = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                      histogram_freq=1) # epoch마다 히스토그램 계산
def scheduler(epoch, lr):
   if epoch < 2:
     return lr
   else:
     return lr * 0.1

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=1, min_lr=0.000001)
# lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)

print(current_time + model_name)
model50.fit(train_generator, epochs=100, validation_data=val_generator,
            callbacks= [es, board, lr_schedule], verbose= 1)

save_dir = 'D:/alpaco/image_classification/models/' + current_time + model_name
# 모델 저장
model50.save(save_dir + '.h5')

log_eval1 = "D:/alpaco/image_classification/logs/eval/" + current_time + model_name
board_eval1 = tf.keras.callbacks.TensorBoard(log_dir=log_eval1, 
                                                      histogram_freq=1) # epoch마다 히스토그램 계산

model50.evaluate(test_generator, callbacks= board_eval1)

base_model50.trainable = True

es = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience = 25, restore_best_weights= True)

current_time = datetime.datetime.now().strftime("%Y%m%d%H%M") 
log_dir = "D:/alpaco/image_classification/logs/fit/" + current_time + model_name
board = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                      histogram_freq=1) # epoch마다 히스토그램 계산

model50.fit(train_generator, epochs=200, validation_data=val_generator,
            callbacks= [es, board, lr_schedule], verbose= 1)

log_eval2 = "D:/alpaco/image_classification/logs/eval/" + current_time + model_name
board_eval2 = tf.keras.callbacks.TensorBoard(log_dir=log_eval2, 
                                                      histogram_freq=1) # epoch마다 히스토그램 계산
model50.evaluate(test_generator, callbacks= board_eval2) 

save_dir = 'D:/alpaco/image_classification/models/' + current_time + model_name
# 모델 저장
model50.save(save_dir + '.h5')
