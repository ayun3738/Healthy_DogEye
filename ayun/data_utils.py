import cv2
import json
import numpy as np

def resize(img, img_size):
  if(img.shape[1] > img.shape[0]) : 
    ratio = img_size/img.shape[1]
  else :
    ratio = img_size/img.shape[0]

  img = cv2.resize(img, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR) # interpolation : 고간법 - 주변을 참고해서 채움

  return img

def padding(img, img_size):
  w, h = img.shape[1], img.shape[0]

  dw = (img_size-w)/2 # img_size와 w의 차이
  dh = (img_size-h)/2 # img_size와 h의 차이

  M = np.float32([[1,0,dw], [0,1,dh]])  #(2*3 이차원 행렬)
  img_re = cv2.warpAffine(img, M, (img_size, img_size))

  return img_re

def select_normal(file):
  with open(file, "r", encoding='UTF8') as st_json:
    dic = json.load(st_json)
    device_type = dic['images']['meta']['device']
    if (device_type == '일반카메라') | (device_type == '스마트폰'):
        return dic['images']['meta']['file_name']
    else:
        return 'nan'
                    