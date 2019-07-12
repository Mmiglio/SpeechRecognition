'''
Models implementations
'''

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


def lstm_att_model(input_shape=(99, 40),
                   cnn_features=30,
                   multi_lstm=True,
                   attention=True):
    '''
    Long-Short-Term-Memory model

    Parameters:\n
    input_shape (array): dimensions of the model input\n
    cnn_features (int): number of features for the first CNN Layer\n
    multi_lstm (bool): activate or deactivate the second LSTM Layer\n
    attention (bool): activate or deactivate the Attention Layer\n

    Returns:\n
    tf.keras.Model: Model built with keras
    '''

    # Fetch input
    inputs = tf.keras.Input(shape=input_shape)
    reshape = tf.keras.layers.Reshape(
        input_shape=input_shape, target_shape=(99, 40, 1))(inputs)

    # Normalization Layer
    layer_out = tf.keras.layers.BatchNormalization()(reshape)

    # Convolutional Layer
    layer_out = tf.keras.layers.Conv2D(cnn_features, kernel_size=(3, 3),
                                       padding='same', activation='relu')(layer_out)
    layer_out = tf.keras.layers.BatchNormalization()(layer_out)
    layer_out = tf.keras.layers.Conv2D(1, kernel_size=(3, 3),
                                       padding='same', activation='relu')(layer_out)
    layer_out = tf.keras.layers.BatchNormalization()(layer_out)
    layer_out = tf.keras.layers.Lambda(
        lambda x: tf.keras.backend.squeeze(x, -1), name='squeeze_dim')(layer_out)

    # LSTM Layer
    layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        60, return_sequences=True, dropout=0.2))(layer_out)
    if multi_lstm:
        layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            60, return_sequences=True, dropout=0.2))(layer_out)

    # Attention Layer
    if attention:
        query, value = tf.keras.layers.Lambda(
            lambda x: tf.split(x, num_or_size_splits=2, axis=2))(layer_out)
        layer_out = tf.keras.layers.Attention()([query, value])

    # Classification Layer
    outputs = tf.keras.layers.Flatten()(layer_out)
    outputs = tf.keras.layers.Dense(512, activation='relu')(outputs)
    outputs = tf.keras.layers.Dense(30, activation='softmax')(outputs)

    # Output Model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model

    # FUTURE GRU BASED MODEL

    # model.add(tf.keras.layers.GRU(60, input_shape=input_shape, dropout=0.2))
    # model.add(tf.keras.layers.Dense(60, activation='relu'))
    # model.add(tf.keras.layers.Dense(30, activation='softmax'))
