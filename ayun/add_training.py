import pandas as pd
import numpy as np
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D
from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout, BatchNormalization
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
import os

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

# 이미지 데이터 제너레이터 생성
datagen = ImageDataGenerator(
    rescale=1./255)

model_name = 'res50'
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M") 


train_path = 'D:/alpaco/image_classification/imgs/dogs/eye_multiclass/dataset/train'
val_path = 'D:/alpaco/image_classification/imgs/dogs/eye_multiclass/dataset/val'
test_path = 'D:/alpaco/image_classification/imgs/dogs/eye_multiclass/test_balanced'

# 이미지 데이터를 불러오기 위한 제너레이터 생성
train_generator = datagen.flow_from_directory(
    directory= train_path,
    target_size=(224, 224),
    batch_size=128,
    shuffle = True,
    class_mode='categorical')
val_generator = datagen.flow_from_directory(
    directory= val_path ,
    target_size=(224, 224),
    batch_size=128,
    shuffle = False,
    class_mode='categorical')
test_generator = datagen.flow_from_directory(
    directory= test_path ,
    target_size=(224, 224),
    batch_size=128,
    shuffle = False,
    class_mode='categorical')

model_base = 'D:/alpaco/image_classification/models'
load_name = '202303252211res50.h5'
model50 = tf.keras.models.load_model(model_base + '/' + load_name)
print(model50.layers)

log_eval1 = "D:/alpaco/image_classification/logs/eval/" + load_name + current_time
board_eval1 = tf.keras.callbacks.TensorBoard(log_dir=log_eval1, 
                                                      histogram_freq=1) # epoch마다 히스토그램 계산

model50.evaluate(test_generator, callbacks= board_eval1)

model50.trainable = True

model50.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                metrics = ['accuracy', tf.keras.metrics.Recall()]
                )

es = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience = 15, restore_best_weights= True)

log_dir = "D:/alpaco/image_classification/logs/fit/" + current_time + model_name
board = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                      histogram_freq=1) # epoch마다 히스토그램 계산

model50.fit(train_generator, epochs=100, validation_data=val_generator,
            callbacks= [es, board], verbose= 1)

current_time = datetime.datetime.now().strftime("%Y%m%d%H%M") 
log_eval2 = "D:/alpaco/image_classification/logs/eval/" + load_name + current_time
board_eval2 = tf.keras.callbacks.TensorBoard(log_dir=log_eval2, 
                                                      histogram_freq=1) # epoch마다 히스토그램 계산
model50.evaluate(test_generator, callbacks= board_eval2) 

save_dir = 'D:/alpaco/image_classification/models/' + current_time + model_name
# 모델 저장
model50.save(save_dir + '.h5')
