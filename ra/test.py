from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, save_model
import tensorflow as tf
import datetime

# 이미지 데이터 제너레이터 생성
test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split = 0.2
)

# 이미지 데이터를 불러오기 위한 제너레이터 생성
test_generator = test_datagen.flow_from_directory(
    directory= './output/train/' ,
    target_size=(224, 224),
    batch_size=256,
    shuffle = True,
    class_mode='categorical')


# 모델로드
model = tf.keras.models.load_model('./model/202303231618res50.h5')



model_name = 'res50'
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M") 
log_dir = "/home/rara/data/log_test/" + current_time + model_name
board = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# test_evaluate
model.evaluate(test_generator,callbacks=[board]) 