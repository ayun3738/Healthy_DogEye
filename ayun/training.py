import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
import os

from deep import deep_cnn
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="0"

label2idx = {'무증상' : 0, '결막염' : 1, '궤양성각막질환' : 2 , '백내장' : 3 , '비궤양성각막질환' : 4,
               '색소침착성각막염' : 5, '안검내반증' : 6, '안검염' : 7, '안검종양' : 8, '유루증' : 9, '핵경화' : 10}
idx2label = {0: '무증상', 1: '결막염', 2: '궤양성각막질환', 3: '백내장', 4: '비궤양성각막질환',
               5: '색소침착성각막염', 6: '안검내반증', 7: '안검염', 8: '안검종양', 9: '유루증', 10: '핵경화'}

base_load = 'D:/alpaco/팀프로젝트_이미지분류/imgs/dogs/eye'
base_load_m = 'D:/alpaco/팀프로젝트_이미지분류/imgs/dogs/eye_mu'
img_1 = pd.read_pickle(base_load + '/결막염_images_224_900.pickle')
img_2 = pd.read_pickle(base_load + '/궤양성각막질환_images_224_900.pickle')
img_3 = pd.read_pickle(base_load + '/백내장_images_224_900.pickle')
img_4 = pd.read_pickle(base_load + '/비궤양성각막질환_images_224_900.pickle')
img_5 = pd.read_pickle(base_load + '/색소침착성각막염_images_224_900.pickle')
img_6 = pd.read_pickle(base_load + '/안검내반증_images_224_900.pickle')
img_7 = pd.read_pickle(base_load + '/안검염_images_224_900.pickle')
img_8 = pd.read_pickle(base_load + '/안검종양_images_224_900.pickle')
img_9 = pd.read_pickle(base_load + '/유루증_images_224_900.pickle')
img_10 = pd.read_pickle(base_load + '/핵경화_images_224_900.pickle')

img_11 = pd.read_pickle(base_load_m + '/결막염_images_224_200.pickle')
img_12 = pd.read_pickle(base_load_m + '/궤양성각막질환_images_224_200.pickle')
img_13 = pd.read_pickle(base_load_m + '/백내장_images_224_200.pickle')
img_14 = pd.read_pickle(base_load_m + '/비궤양성각막질환_images_224_200.pickle')
img_15 = pd.read_pickle(base_load_m + '/색소침착성각막염_images_224_200.pickle')
img_16 = pd.read_pickle(base_load_m + '/안검내반증_images_224_200.pickle')
img_17 = pd.read_pickle(base_load_m + '/안검염_images_224_200.pickle')
img_18 = pd.read_pickle(base_load_m + '/안검종양_images_224_200.pickle')
img_19 = pd.read_pickle(base_load_m + '/유루증_images_224_200.pickle')
img_20 = pd.read_pickle(base_load_m + '/핵경화_images_224_200.pickle')

img = np.concatenate((img_1, img_2, img_3, img_4, img_5, img_6, img_7, img_8, img_9, img_10,
                      img_11, img_12, img_13, img_14, img_15, img_16, img_17, img_18, img_19, img_20))
print('img_shape : ', img.shape)

# np.full(num, a) -> a로 num개만큼 채운 numpy 리스트
label = np.concatenate((np.full(900, 1), np.full(900, 2), np.full(900, 3), np.full(900, 4), np.full(900, 5),
                        np.full(900, 6), np.full(900, 7), np.full(900, 8), np.full(900, 9), np.full(900, 10), np.full(1000, 0)))
print('label_shape : ', label.shape)


X_train, X_test, y_train, y_test = train_test_split(img, label, train_size= 0.8)

model = deep_cnn(X_train.shape)

model_name = 'deep_ver' + datetime.datetime.now().strftime("%m%d%H%M")
current_time = datetime.datetime.now().strftime("%Y%m%d") 
print(model_name)

log_dir = "D:/alpaco/팀프로젝트_이미지분류/logs/fit/" + current_time + model_name
board = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                      histogram_freq=1) # epoch마다 히스토그램 계산
es = tf.keras.callbacks.EarlyStopping(patience= 15, restore_best_weights= True)
# batch = 64, epoch = 300, patience = 15, 
model.fit(X_train, y_train, batch_size= 32, validation_split = 0.2, epochs = 300, verbose = 1,
          callbacks=[board, es])

model_save = 'D:/alpaco/팀프로젝트_이미지분류/models'
model.save(model_save + '/' + model_name + '.h5')