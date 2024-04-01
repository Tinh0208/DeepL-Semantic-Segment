from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization

def segnet(input_shape, n_classes):
    # Encoder
    inputs = Input(shape=input_shape)

    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_2)

    conv_3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_4)

    conv_5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_6)

    conv_7 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_3)
    conv_7 = BatchNormalization()(conv_7)
    conv_8 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv_7)
    conv_8 = BatchNormalization()(conv_8)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(conv_8)

    # Decoder
    up_1 = UpSampling2D(size=(2, 2))(pool_4)
    conv_9 = Conv2D(512, (3, 3), activation='relu', padding='same')(up_1)
    conv_9 = BatchNormalization()(conv_9)
    conv_10 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv_9)
    conv_10 = BatchNormalization()(conv_10)

    up_2 = UpSampling2D(size=(2, 2))(conv_10)
    conv_11 = Conv2D(256, (3, 3), activation='relu', padding='same')(up_2)
    conv_11 = BatchNormalization()(conv_11)
    conv_12 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_11)
    conv_12 = BatchNormalization()(conv_12)

    up_3 = UpSampling2D(size=(2, 2))(conv_12)
    conv_13 = Conv2D(128, (3, 3), activation='relu', padding='same')(up_3)
    conv_13 = BatchNormalization()(conv_13)
    conv_14 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_13)
    conv_14 = BatchNormalization()(conv_14)

    up_4 = UpSampling2D(size=(2, 2))(conv_14)
    conv_15 = Conv2D(64, (3, 3), activation='relu', padding='same')(up_4)
    conv_15 = BatchNormalization()(conv_15)
    conv_16 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_15)
    conv_16 = BatchNormalization()(conv_16)

    # Output
    output = Conv2D(n_classes, (1, 1), activation='softmax')(conv_16)

    model = Model(inputs=[inputs], outputs=[output])

    return model
