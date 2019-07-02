import tensorflow as tf

def cnnModel(input_shape=(99, 40)):
    """
    Model consisting of 4 convolution blocks
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
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    # linear
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    # softmax
    model.add(tf.keras.layers.Dense(30, activation='softmax'))

    return model


def rnnModel(input_shape=(99, 40)):
    '''
    Long-Short-Term-Memory model
    '''
    
    # Model Base
    model = tf.keras.models.Sequential()

    # Normalization Layer
    model.add(tf.keras.layers.BatchNormalization(input_shape=input_shape))

    # LSTM Layer with CPU
    # model.add(
    #     tf.keras.layers.Bidirectional(
    #         tf.keras.layers.LSTM(60, return_sequences=True, dropout=0.2)
    #     )
    # )
    # LSTM Layer with GPU
    model.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.CuDNNLSTM(60, return_sequences=True, dropout=0.2)
        )
    )

    # Classification Layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(30, activation='softmax'))

    return model

    # FUTURE GRU BASED MODEL

    # model.add(tf.keras.layers.GRU(60, input_shape=input_shape, dropout=0.2))
    # model.add(tf.keras.layers.Dense(60, activation='relu'))
    # model.add(tf.keras.layers.Dense(30, activation='softmax'))
