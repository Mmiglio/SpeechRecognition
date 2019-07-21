import tensorflow as tf


def cnnModel(input_shape=(99, 40)):
    """
    Model consisting of 4 convolution blocks. 1.2M parameters
    Accuracy = 0.96
    """

    model = tf.keras.models.Sequential()

    # Normalization layer
    model.add(tf.keras.layers.Reshape(input_shape=input_shape, target_shape=(99, 40, 1)))
    model.add(tf.keras.layers.BatchNormalization())

    filters = [16, 32, 64, 128]

    for num_filters in filters:
        # Conv a
        model.add(tf.keras.layers.Conv2D(
            num_filters,
            kernel_size=(3, 3),
            padding='same'
            )
        )
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))

        # Conv b
        model.add(tf.keras.layers.Conv2D(
            num_filters,
            kernel_size=(3, 3),
            padding='same'
            )
        )
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))

        # Pooling
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.2))

    # Classification layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(30, activation='softmax'))

    return model


def smallCnnModel(input_shape=(99, 40)):
    """
    Model with 150k parameters.
    Accuracy = 0.95
    """

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Reshape(input_shape=input_shape, target_shape=(99, 40, 1)))

    model.add(tf.keras.layers.Convolution2D(32, (1, 10), padding='same', activation='relu'))
    model.add(tf.keras.layers.Convolution2D(64, (1, 5), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((1, 4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Convolution2D(64, (1, 10), padding='valid', activation='relu'))
    model.add(tf.keras.layers.Convolution2D(128, (10, 1), padding='same', activation='relu'))

    model.add(tf.keras.layers.GlobalMaxPooling2D())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(128, activation='relu', name='FEATURES'))
    model.add(tf.keras.layers.Dense(30, activation='softmax'))

    return model


def modular_cnn_model(input_shape=(99, 40)):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Reshape(
        input_shape=input_shape, target_shape=(input_shape[0], input_shape[1], 1)))

    model.add(tf.keras.layers.Convolution2D(
        32, (1, 10), padding='same', activation='relu'))
    model.add(tf.keras.layers.Convolution2D(
        64, (1, 5), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((1, 4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Convolution2D(
        64, (1, 10), padding='valid', activation='relu'))
    model.add(tf.keras.layers.Convolution2D(
        128, (10, 1), padding='same', activation='relu'))

    model.add(tf.keras.layers.GlobalMaxPooling2D())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(128, activation='relu', name='FEATURES'))
    model.add(tf.keras.layers.Dense(30, activation='softmax'))

    return model