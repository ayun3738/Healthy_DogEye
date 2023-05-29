import os
import json
import shutil
import cv2
import numpy as np
import pickle

from data_utils import resize, padding, select_normal

# 가상환경 /alpaco/vscode/testenv/Scripts/activate.bat

# 안구, 일반카메라, img 데이터 정리
# base_load = 'D:/alpaco/팀프로젝트_이미지분류/153.반려동물 안구질환 데이터/01.데이터/1.Training/라벨링데이터/TL1/개/안구/일반'
# base_save = 'D:/alpaco/image_classification/imgs/dogs/eye_binary'

# 결막염 : conjunctivitis
# 궤양각막 : corneal_ulcer
# 비궤양각막 : corneal
# 백내장 : Cataract
# 색소침착성 각막염 : PIH
# 안검염 : blepharitis
# 안검종양 : Xanthelasma
# 안검내반증 : Entropion
# 유루증 : epiphora
# 핵경화 : Nuclear_Sclerosis


base = 'D:/alpaco/image_classification/imgs/dogs/eye_binary/test'
label_names = os.listdir(base)
IMG_SIZE = 224

for label_name in label_names:
    print(f'{label_name} start')

    folder_paths = base + '/' + label_name
    folder_names = os.listdir(folder_paths)
    for folder_name in folder_names:
    
        folder_path = folder_paths + '/' + folder_name
        files = os.listdir(folder_path)
        for file in files:
            
            full_path = folder_path + '/' + file
            # img_array = np.fromfile(full_path, np.uint8)
            # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            img = cv2.imread(full_path)
            img_re = resize(img, IMG_SIZE)
            img_pre = padding(img_re, IMG_SIZE)
            
            cv2.imwrite(full_path, img_pre)
            
    print(f'{label_name} end')

