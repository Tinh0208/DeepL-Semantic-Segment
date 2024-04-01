import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization


# Định nghĩa mô hình FCN
def fcn(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)

    # Decoder
    up1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4)
    up1 = BatchNormalization()(up1)
    up1 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
    up1 = BatchNormalization()(up1)
    merge1 = tf.keras.layers.concatenate([conv3, up1], axis=-1)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge1)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5)
    up2 = BatchNormalization()(up2)
    up2 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    up2 = BatchNormalization()(up2)
    merge2 = tf.keras.layers.concatenate([conv2, up2], axis=-1)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge2)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    up3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6)
    up3 = BatchNormalization()(up3)
    up3 = Conv2D(32, (3, 3), activation='relu', padding='same')(up3)
    up3 = BatchNormalization()(up3)
    merge3 = tf.keras.layers.concatenate([conv1, up3], axis=-1)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge3)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    # Output layer
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(conv7)

    model = Model(inputs=inputs, outputs=outputs)
    return model

