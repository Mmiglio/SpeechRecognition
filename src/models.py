import tensorflow as tf


def cnnModel(input_shape=(99, 40)):
    """
    Simple model consisting of 4 blocks
    """

    model = tf.keras.models.Sequential()

    # Normalization layer
    model.add(tf.keras.layers.Reshape(input_shape=input_shape, target_shape=(99, 40, 1)))
    model.add(tf.keras.layers.BatchNormalization())

    ###########################
    # Conv 1a
    model.add(tf.keras.layers.Conv2D(
        16,
        kernel_size=(3, 3),
        padding='same'
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    # Conv 1b
    model.add(tf.keras.layers.Conv2D(
        16,
        kernel_size=(3, 3),
        padding='same'
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    # Pooling 1
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    ###########################
    # Conv 2a
    model.add(tf.keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        padding='same'
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    # Conv 2b
    model.add(tf.keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        padding='same'
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    # Pooling 2
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    ###########################
    # Conv 3a
    model.add(tf.keras.layers.Conv2D(
        64,
        kernel_size=(3, 3),
        padding='same'
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    # Conv 3b
    model.add(tf.keras.layers.Conv2D(
        64,
        kernel_size=(3, 3),
        padding='same'
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    # Pooling 3
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    ###########################
    # Conv 4a
    model.add(tf.keras.layers.Conv2D(
        128,
        kernel_size=(3, 3),
        padding='same'
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    # Conv 4b
    model.add(tf.keras.layers.Conv2D(
        128,
        kernel_size=(3, 3),
        padding='same'
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    # Pooling 4
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    # Classification
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    # linear
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    # softmax
    model.add(tf.keras.layers.Dense(30, activation='softmax'))

    return model
