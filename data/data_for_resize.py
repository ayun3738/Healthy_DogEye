import os
import json
import shutil

# 안구, 일반카메라, img 데이터 정리
medical_type = '안검내반증'
y_n = '유'
print(medical_type, y_n)
base_load = 'C:/alpaco/팀프로젝트_이미지분류/153.반려동물 안구질환 데이터/01.데이터/1.Training/원천데이터/TS2/개/안구/일반/'
base_save = 'C:/alpaco/팀프로젝트_이미지분류/imgs/dogs/eye/'
load_dir_path = base_load + medical_type + '/' + y_n
save_dir_path = base_save + medical_type + '/' + y_n

file_list = os.listdir(load_dir_path)
jpgs = []
pngs = []
jsons = []
names = []
for file in file_list:
    f_s = file.split('.')
    extension = f_s[-1]
    if extension == 'json':
        jsons.append(file)
        names.append(f_s[0])
    elif extension == 'jpg':
        jpgs.append(file)
    elif extension == 'png':
        pngs.append(file)

normals = []
normals_img = []
gums = []

for j in jsons:
    j_path = load_dir_path + '/' + j
    with open(j_path, "r", encoding='UTF8') as st_json:
        dic = json.load(st_json)
        device_type = dic['images']['meta']['device']
        if (device_type == '일반카메라') | (device_type == '스마트폰'):
            normals.append(j)
            file_name = dic['images']['meta']['file_name']
            normals_img.append(file_name)

        # elif device_type == '검안경':
        #     gums.append(j.split('.')[0])

print(len(jsons))
print(len(normals))
print(len(normals_img))

for img, json in zip(normals_img, normals):
    shutil.move(load_dir_path + '/' + img, save_dir_path + '/' + img)
    shutil.move(load_dir_path + '/' + json, save_dir_path + '/jsons/' + json)
    