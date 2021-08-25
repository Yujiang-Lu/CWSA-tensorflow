import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend, layers, models, utils, Model
from efficientnet.model import EfficientNetB0
import cv2
import numpy as np

tf.compat.v1.disable_eager_execution()

def gen_bal(val_size, frame_length, val_real_path, val_fake_path):
    # real_path = r'G:/data/train/real/c40/val_gen_bal/'
    # fake_path = r'G:/data/train/fake/c40/val_gen_bal/'
    real_path = val_real_path
    fake_path = val_fake_path

    real_pths = [real_path + i for i in os.listdir(real_path)]
    fake_pths = [fake_path + i for i in os.listdir(fake_path)]

    reals = []
    fakes = []

    for i in real_pths:
        files = os.listdir(i)
        files.sort(key=lambda x: int(x[:-4]))
        reals.append([i + '/' + file for file in files])
    for i in fake_pths:
        files = os.listdir(i)
        files.sort(key=lambda x: int(x[:-4]))
        fakes.append([i + '/' + file for file in files])
    files = reals + fakes
    print('Validation data length : ', len(files))
    data = []
    label = []
    path = []
    for i in range(val_size):
        sample = []
        sample_index = random.randint(0, len(files)-1)
        video_length = len(files[sample_index])
        start_index = random.randint(0, video_length-1-frame_length)
        for j in range(start_index, start_index+frame_length):
            img = cv2.imread(files[sample_index][j])
            if img is not None:
                img = img / 255.0
            sample.append(img)
        data.append(sample)
        if 'real' in files[sample_index][0]:
            label.append(0)
        else:
            label.append(1)
        path.append(files[sample_index][start_index])

    return np.array(data), np.array(label), np.array(path)


model = models.load_model('./pre_model_EfficientNetB0_CWSA/pre_model/model_save')
print(model.summary())

frame_length = 5
real_path = r'G:/data/train/real/c40/val_gen_bal/'
fake_path = r'G:/data/train/fake/c40/val_gen_bal/'
x, y, z = gen_bal(2000, frame_length, val_fake_path=fake_path, val_real_path=real_path)

sum_count = 0
right_count = 0
failure_path = []
for i, frame in enumerate(x):
    y_pre = model.predict([[frame]])
    sum_count += 1
    if y_pre < 0.5 and y[i] == 0:
        right_count += 1
    elif y_pre >= 0.5 and y[i] == 1:
        right_count += 1
    else:
        failure_path.append(z[i])
print("accuracy: ", right_count / sum_count)

wrong_path = r'C:/Users/Administrator/Desktop/wrong_example_2'
for i, path in enumerate(failure_path):
    img = cv2.imread(path)
    cv2.imwrite(wrong_path + '/' + str(i) + '.png', img)
    print(str(i) + '.png', '  <-  ', path)


