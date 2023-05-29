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
# base_load = 'D:/alpaco/팀프로젝트_이미지분류/153.반려동물 안구질환 데이터/01.데이터/1.Training/라벨링데이터/TL1/개/안구/일반'
# label_names = os.listdir(base_load)
# IMG_SIZE = 224
# NUM_SAMPLES = 1800

# jsons = []

# for label_name in label_names:

#     folder_paths = base_load + '/' + label_name
#     folder_names = os.listdir(folder_paths)
#     for folder_name in folder_names:
#         # 유 or 무 결정
#         folder_path = folder_paths + '/' + folder_name
#         files = os.listdir(folder_path)
#         # for file in files[:NUM_SAMPLES]:
#         for file in files:
#             if file.split('.')[-1] == 'json':
#                 jsons.append(file)    


# base_load = 'D:/alpaco/팀프로젝트_이미지분류/153.반려동물 안구질환 데이터/01.데이터/1.Training/라벨링데이터/TL2/개/안구/일반'
# label_names = os.listdir(base_load)
# IMG_SIZE = 224
# NUM_SAMPLES = 1800

# for label_name in label_names:

#     folder_paths = base_load + '/' + label_name
#     folder_names = os.listdir(folder_paths)
#     for folder_name in folder_names:
#         # 유 or 무 결정
#         folder_path = folder_paths + '/' + folder_name
#         files = os.listdir(folder_path)
#         # for file in files[:NUM_SAMPLES]:
#         for file in files:
#             if file.split('.')[-1] == 'json':
#                 jsons.append(file)    


# print(len(jsons))
# print(len(set(jsons)))

import splitfolders

base = 'D:/alpaco/image_classification/imgs/dogs/eye_multiclass/'

splitfolders.ratio(base + 'train_balanced', output=base + 'dataset', seed=77, ratio=(0.8, 0.2))
