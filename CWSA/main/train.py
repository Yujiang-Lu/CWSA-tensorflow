import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend, layers, models, utils, Model
from efficientnet.model import EfficientNetB0
import cv2
import numpy as np

tf.compat.v1.disable_eager_execution()

def feature_reset(x):
    x = backend.permute_dimensions(x, (0, 4, 2, 3, 1))
    return x

def attention(x):
    x1 = backend.mean(x, axis=-1, keepdims=True)
    x2 = backend.mean(x1, axis=-2, keepdims=True)
    x3 = backend.mean(x2, axis=-3, keepdims=True)

    x3 = backend.repeat_elements(x3, 7, 2)
    x3 = backend.repeat_elements(x3, 7, 3)
    x3 = backend.repeat_elements(x3, 5, 4)
    x3 = backend.sigmoid(x3)
    return x3

def model(FRAME_NUM):
    frame_num = FRAME_NUM
    height = 224
    width = 224
    channel = 3
    base = EfficientNetB0(input_shape=(224, 224, 3),
                          include_top=False,
                          weights='imagenet',
                          backend=backend,
                          layers=layers,
                          models=models,
                          utils=utils)
    base_ = Model(base.inputs, [base.get_layer('block3a_expand_activation').output,
                                base.get_layer('block4a_expand_activation').output,
                                base.get_layer('block6a_expand_activation').output,
                                base.get_layer('top_activation').output])
    frame_input = layers.Input((height, width, channel))
    f56, f28, f14, f7 = base_(frame_input)
    f56 = layers.AveragePooling2D((8, 8), strides=8)(f56)
    f28 = layers.AveragePooling2D((4, 4), strides=4)(f28)
    f14 = layers.AveragePooling2D((2, 2), strides=2)(f14)
    f_total = layers.Concatenate()([f56, f28, f14, f7])
    f_total = layers.Conv2D(filters=128, kernel_size=(1, 1))(f_total)
    f_total = layers.BatchNormalization()(f_total)
    f_total = layers.LeakyReLU(0.0)(f_total)
    base = Model(frame_input, f_total)

# CWSA model shown below

    # no multi-scale
    # base = EfficientNetB0(input_shape=(224, 224, 3),
    #                       include_top=False,
    #                       weights='imagenet',
    #                       backend=backend,
    #                       layers=layers,
    #                       models=models,
    #                       utils=utils)

    # CWSA
    # seq_input = layers.Input((frame_num, height, width, channel))
    # x = layers.TimeDistributed(base)(seq_input)
    #
    # x = layers.Lambda(feature_reset)(x)
    # att = layers.Lambda(attention)(x)
    # x = layers.Multiply()([att, x])
    #
    # x = layers.TimeDistributed(layers.BatchNormalization())(x)
    # x = layers.TimeDistributed(layers.Conv2D(filters=64, kernel_size=(3, 3)))(x)
    # x = layers.TimeDistributed(layers.BatchNormalization())(x)
    # x = layers.LeakyReLU(0.1)(x)
    # x = layers.TimeDistributed(layers.Conv2D(filters=64, kernel_size=(3, 3)))(x)
    # x = layers.LeakyReLU(0.1)(x)
    # x = layers.TimeDistributed(layers.Conv2D(filters=1, kernel_size=(3, 3)))(x)
    # # x = layers.LeakyReLU(0)(x)
    # x = layers.Flatten()(x)

    # LSTM
    # seq_input = layers.Input((frame_num, height, width, channel))
    # x = layers.TimeDistributed(base)(seq_input)
    # x = layers.Reshape((5, 7*7*1280))(x)
    # x = layers.LSTM(units=128)(x)
    # x = layers.Flatten()(x)

    # traditional RNN
    # seq_input = layers.Input((frame_num, height, width, channel))
    # x = layers.TimeDistributed(base)(seq_input)
    # x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    # x = layers.Reshape((5, 1280))(x)
    # x = layers.RNN(layers.SimpleRNNCell(128), return_sequences=True, return_state=True)(x)
    # x = layers.Flatten()(x[0])

    # simple average fusion
    # seq_input = layers.Input((frame_num, height, width, channel))
    # x = layers.TimeDistributed(base)(seq_input)
    # x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    # x = tf.reduce_mean(x, 1)
    #
    # x = layers.Flatten()(x)

    seq_input = layers.Input((frame_num, height, width, channel))
    x = layers.TimeDistributed(base)(seq_input)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model_ = Model(inputs=seq_input, outputs=x, name='LYJnet')
    model_.summary()
    return model_

def gen(batch_size, frame_length, train_real_path, train_fake_path):
    real_path = train_real_path
    fake_path = train_fake_path

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
        reals.append([i + '/' + file for file in files])
    files = reals + fakes
    print('Train data length : ', len(files))
    while True:
        data = []
        label = []
        for i in range(batch_size):
            sample = []
            sample_index = random.randint(0, len(files)-1)
            video_length = len(files[sample_index])
            start_index = random.randint(0, video_length-1-frame_length)
            for j in range(start_index, start_index + frame_length):
                img = cv2.imread(files[sample_index][j])
                if img is not None:
                    img = img/255.0
                sample.append(img)
            data.append(sample)
            if 'real' in files[sample_index][0]:
                label.append(0)
            else:
                label.append(1)
        yield np.array(data), np.array(label)


def gen_bal(val_size, frame_length, val_real_path, val_fake_path):
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
    return np.array(data), np.array(label)


train_fake_path = r'I:/FF++_c40_frame_1.5/fake/NeuralTextures/c40/train/'
val_fake_path = r'I:/FF++_c40_frame_1.5/fake/NeuralTextures/c40/val/'
train_real_path = r'I:/FF++_c40_frame_1.5/real/Real/c40/train/'
val_real_path = r'I:/FF++_c40_frame_1.5/real/Real/c40/val/'

train_gen = gen(10, 1, train_fake_path=train_fake_path, train_real_path=train_real_path)
model = model(1)
x, y = gen_bal(1000, 1, val_fake_path=val_fake_path, val_real_path=val_real_path)


model.compile(loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'],
              optimizer=tf.keras.optimizers.SGD(0.01, momentum=0.9))

history = model.fit(train_gen,
                    steps_per_epoch=300,
                    validation_data=(x, y),
                    epochs=40,
                    verbose=1
                    )

# model.compile(loss=tf.keras.losses.binary_crossentropy,
#               metrics=['accuracy'],
#               optimizer=tf.keras.optimizers.SGD(0.01, momentum=0.9))
#
#
# history = model.fit(train_gen,
#                     steps_per_epoch=300,
#                     validation_data=(x, y),
#                     validation_steps=100,
#                     epochs=30,
#                     verbose=1)

models.save_model(model, filepath='./pre_model/model_save')

image_generate = True

if(image_generate == True):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.3, 1])
    plt.legend(loc='lower right')
    plt.savefig(fname="ACC_convergence.png", figsize=[10, 10])
    plt.show()

