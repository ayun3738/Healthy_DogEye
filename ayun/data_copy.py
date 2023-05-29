import shutil
import os

base_load = 'D:/alpaco/팀프로젝트_이미지분류/153.반려동물 안구질환 데이터/01.데이터/1.Training/라벨링데이터/TL1/개/안구/일반'
base_save = 'D:/alpaco/팀프로젝트_이미지분류/imgs/dogs/eye'
label_names = os.listdir(base_load)
IMG_SIZE = 224


for label_name in label_names:
    label_dic = {}

    folder_paths = base_load + '/' + label_name
    folder_names = os.listdir(folder_paths)
    for folder_name in folder_names:
         if folder_name != '무':
            folder_path = folder_paths + '/' + folder_name
            files = os.listdir(folder_path)
            for file in files:
                if file.split('.')[-1] != 'json':
                    
                    full_path = folder_path + '/' + file
                    img_array = np.fromfile(full_path, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    img_re = resize(img, IMG_SIZE)
                    img_pre = padding(img_re, IMG_SIZE)
                    
                    images.append(img_pre)
                    labels.append(label2index[label_name])
            
    print(f'라벨 : {label_name}')
    print(len(images), len(labels))

images = np.array(images)
labels = np.array(labels)
print(images.shape, labels.shape)
shutil.copy()