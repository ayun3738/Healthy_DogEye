import os
import json
import shutil
import cv2
import numpy as np
import pickle

from data_utils import resize, padding, select_normal

# 가상환경 /alpaco/vscode/testenv/Scripts/activate.bat

# 안구, 일반카메라, img 데이터 정리
# base_load = 'D:/alpaco/팀프로젝트_이미지분류/153.반려동물 안구질환 데이터/01.데이터/2.Validation/라벨링데이터/VL/개/안구/일반'
base_load = 'D:/alpaco/팀프로젝트_이미지분류/153.반려동물 안구질환 데이터/01.데이터/1.Training/라벨링데이터/TL2/개/안구/일반'
base_save = 'D:/alpaco/팀프로젝트_이미지분류/imgs/dogs/eye'
label_names = os.listdir(base_load)
IMG_SIZE = 224
NUM_SAMPLES = 1800

print(base_load.split('/')[-4], base_save.split('/')[-1])

label2index = {'결막염' : 0, '궤양성각막질환' : 1 , '백내장' : 2 , '비궤양성각막질환' : 3, '색소침착성각막염' : 4,
               '안검내반증' : 5, '안검염' : 6, '안검종양' : 7, '유루증' : 8, '핵경화' : 9}

for label_name in label_names:

    folder_paths = base_load + '/' + label_name
    folder_names = os.listdir(folder_paths)
    for folder_name in folder_names:
        # 유 or 무 결정
        images = []
        if folder_name != '무':
            folder_path = folder_paths + '/' + folder_name
            files = os.listdir(folder_path)
            # for file in files[:NUM_SAMPLES]:
            for file in files:
                if file.split('.')[-1] != 'json':
                    
                    full_path = folder_path + '/' + file
                    img_array = np.fromfile(full_path, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    img_re = resize(img, IMG_SIZE)
                    img_pre = padding(img_re, IMG_SIZE)
                    
                    images.append(img_pre)
            
            print(f'라벨 : {label_name}_{folder_name}')
            print(len(images))

            images = np.array(images)
            print(images.shape)

            # with open(base_save+ '/' + label_name +'_images_' + str(IMG_SIZE) + '_' + str(images.shape[0]) + '.pickle', 'wb') as f:
            with open(base_save+ '/' + label_name +'_images_' + folder_name + str(IMG_SIZE) + '.pickle', 'wb') as f:
                pickle.dump(images, f)