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

# base = 'D:/alpaco/image_classification/imgs/dogs/eye_binary/test_balanced/cataract'
# base = 'D:/alpaco/image_classification/imgs/dogs/eye_binary/test_balanced/corneal_ulcer'
# base = 'D:/alpaco/image_classification/imgs/dogs/eye_binary/test_balanced/corneal'

# folder_names = os.listdir(base)

# length_m = 0

# for folder_name in folder_names:
#     folder_path = base + '/' + folder_name
#     files = os.listdir(folder_path)
#     if folder_name == 'm':
#         length_m = len(files)
#         print(f'{folder_name} : {length_m}')
#     elif folder_name == 'u':
#         for file in files[length_m:]:
#             os.remove(folder_path + '/' + file)
#         print('u delete')



base = 'D:/alpaco/image_classification/imgs/dogs/eye_multiclass/test_balanced'
# base = 'D:/alpaco/image_classification/imgs/dogs/eye_multiclass/m'
folders = os.listdir(base)
for folder in folders:

    files = os.listdir(base + '/' + folder)
    for file in files[576:]:
        os.remove(base + '/' + folder + '/' + file)