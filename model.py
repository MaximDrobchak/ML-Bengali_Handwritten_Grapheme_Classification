from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Input, SeparableConv2D, GlobalAveragePooling2D, Dense, MaxPooling2D, Activation, Flatten
from tensorflow.keras.layers import add as add_concat
from keras import backend as K
from constants import strategy, bnmomemtum, SHAPE
with strategy.scope():
    def fire(x, filters, kernel_size):
        if not isinstance(filters, list):
            filters = [filters, filters]
        x = SeparableConv2D(filters[0], kernel_size, padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis, center=True, scale=False, momentum=bnmomemtum)(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(filters[1], kernel_size, padding='same', use_bias=False)(x)
        return BatchNormalization(axis=channel_axis, center=True, scale=False, momentum=bnmomemtum)(x)

    def fire_module_separable_conv(filters, kernel_size=(3, 3)):
        return lambda x: fire(x, filters, kernel_size)

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    img_input = Input(shape=SHAPE)

    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
    x = BatchNormalization(axis=channel_axis, center=True, scale=False, momentum=bnmomemtum)(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(axis=channel_axis, center=True, scale=False, momentum=bnmomemtum)(x)
    x = Activation('relu')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis, center=True, scale=False, momentum=bnmomemtum)(residual)

    x = fire_module_separable_conv(128)(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = add_concat([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis, center=True, scale=False, momentum=bnmomemtum)(residual)

    x = Activation('relu')(x)
    x = fire_module_separable_conv(256)(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = add_concat([x, residual])

    for i in range(4):
        residual = x

        x = Activation('relu')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis, center=True, scale=False, momentum=bnmomemtum)(x)
        x = Activation('relu')(x)
        x = fire_module_separable_conv(256)(x)

        x = add_concat([x, residual])


    x = fire_module_separable_conv([728, 1024])(x)
    x = Activation('relu')(x)
    y = GlobalAveragePooling2D()(x)

    y = Dense(3096)(y)
    y = Activation('relu')(y)
    y = Dropout(0.3)(y)

    y = Dense(1548)(y)
    y = Activation('relu')(y)
    y = Dropout(0.3)(y)

    head_root = Dense(168, activation = 'softmax', name='head_root')(y)
    head_vowel = Dense(11, activation = 'softmax', name='head_vowel')(y)
    head_consonant = Dense(7, activation = 'softmax', name='head_consonant')(y)

    model = Model(inputs=img_input, outputs=[head_root, head_vowel, head_consonant])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
