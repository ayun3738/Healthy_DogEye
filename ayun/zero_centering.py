import cv2
import pickle
import numpy as np
import os

train_path = 'D:/alpaco/image_classification/imgs/dogs/eye_multiclass_notm/dataset/train'


img_paths = []
for label in os.listdir(train_path):
  sub_path = train_path + '/' + label + '/'
  for files in os.listdir(sub_path):
    img_paths.append(sub_path + files)


X_train = []

for img_file in img_paths:
  img = cv2.imread(img_file)
  X_train.append(img)

compute_mean = lambda imgs: np.mean(imgs, axis = 0)
mean_img = compute_mean(X_train)
mean_img = mean_img.astype(int)
with open(train_path + '/mean_img.pickle', 'wb') as f:
    pickle.dump(mean_img, f)
