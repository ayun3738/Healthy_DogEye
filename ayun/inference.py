import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


class_names = {0: '무증상', 1: '결막염', 2: '궤양성각막질환', 3: '백내장', 4: '비궤양성각막질환',
               5: '색소침착성각막염', 6: '안검내반증', 7: '안검염', 8: '안검종양', 9: '유루증', 10: '핵경화'}

# 모델 로드
#model = tf.keras.models.load_model('model_res50.h5')
path = '/content/drive/MyDrive/image_recognition/image_recofnition_Crawling/raheo/2/'
# 이미지 전처리
for i in range(1,10):
  img = load_img(path + f"강아지 안검종양{i}" , target_size=(224, 224))
  img_array = img_to_array(img)
  img_array = img_array.astype('float32') / 255.0
  img_array = np.expand_dims(img_array, axis=0)

  # 추론
  predictions = model_res50.predict(img_array)


  # 예측 클래스 결정
  predicted_class = np.argmax(predictions, axis=-1)
  class_name = class_names[predicted_class[0]]

  # 출력
  print(f'예측된 클래스: {class_name}인 것 같습니다')
  print(f'클래스별 확률: {predictions[0]}')