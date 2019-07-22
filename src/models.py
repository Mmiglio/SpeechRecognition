'''
Models implementations
'''

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
    model.add(tf.keras.layers.Dense(512, name='features512'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(256, name='features256'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
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


def Inception(input_shape=(99, 40)):
    """
    Neural network containing inception modules
    Accuracy = 0.955
    """

    def inceptionModule(x_input, n):
        """
        Full inception module.
        Graphic representation: https://i.stack.imgur.com/vGIfJ.png
        """
        # Conv 1x1
        conv_1x1 = tf.keras.layers.Conv2D(n, (1, 1), padding='same', activation='relu')(x_input)

        # Conv 3x3
        conv_3x3 = tf.keras.layers.Conv2D(n, (1, 1), padding='same', activation='relu')(x_input)
        conv_3x3 = tf.keras.layers.Conv2D(n, (3, 3), padding='same', activation='relu')(conv_3x3)

        # Conv 5x5
        conv_5x5 = tf.keras.layers.Conv2D(n, (1, 1), padding='same', activation='relu')(x_input)
        conv_5x5 = tf.keras.layers.Conv2D(n, (3, 3), padding='same', activation='relu')(conv_5x5)

        # pool + proj
        pool = tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x_input)
        pool = tf.keras.layers.Conv2D(n, (1, 1), padding='same', activation='relu')(pool)

        output = tf.keras.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool], axis=3)

        return output

    input_layer = tf.keras.layers.Input(input_shape)

    reshape_layer = tf.keras.layers.Reshape(input_shape=input_shape, target_shape=(99, 40, 1))(input_layer)

    x = tf.keras.layers.Conv2D(32, (6, 4), padding='same', strides=(2, 2), activation='relu')(reshape_layer)
    x = tf.keras.layers.MaxPooling2D((3, 2), padding='same', strides=(1, 1))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = inceptionModule(x, 32)
    x = inceptionModule(x, 64)
    x = tf.keras.layers.MaxPooling2D((3, 2))(x)

    x = inceptionModule(x, 64)
    x = inceptionModule(x, 128)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(256, activation='relu')(x)

    x = tf.keras.layers.Dense(30, activation='softmax')(x)

    model = tf.keras.models.Model(input_layer, x)
    return model


def resNet(input_shape=(99, 40)):
    """
    Neural network with residual blocks
    Accuracy = 0.96
    """
    def residualModule(layer_in, n_filters):
        """
        Residual block
        """
        merge_input = layer_in
        # check if the number of filters needs to be increased
        if layer_in.shape[-1] != n_filters:
            merge_input = tf.keras.layers.Conv2D(
                n_filters, (1, 1), padding='same',
                activation='relu',
                kernel_initializer='he_normal')(layer_in)

        conv1 = tf.keras.layers.Conv2D(
            n_filters, (3, 3), padding='same',
            activation='relu',
            kernel_initializer='he_normal')(layer_in)

        conv2 = tf.keras.layers.Conv2D(
            n_filters, (3, 3), padding='same',
            activation='linear',
            kernel_initializer='he_normal')(conv1)

        # add filters
        layer_out = tf.keras.layers.add([conv2, merge_input])
        layer_out = tf.keras.layers.BatchNormalization()(layer_out)
        layer_out = tf.keras.layers.Activation('relu')(layer_out)

        return layer_out

    input_layer = tf.keras.layers.Input(input_shape)

    reshape_layer = tf.keras.layers.Reshape(input_shape=input_shape, target_shape=(99, 40, 1))(input_layer)

    x = tf.keras.layers.Conv2D(64, (6, 4), padding='same', strides=(2, 2), activation='relu')(reshape_layer)
    x = tf.keras.layers.MaxPooling2D((3, 2), padding='same', strides=(1, 1))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = residualModule(x, 64)
    x = residualModule(x, 128)
    x = tf.keras.layers.MaxPooling2D((3, 2))(x)

    x = residualModule(x, 128)
    x = residualModule(x, 256)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)

    x = tf.keras.layers.Dense(30, activation='softmax')(x)

    model = tf.keras.models.Model(input_layer, x, name='Resnet')
    return model


def rnn_att_model(input_shape=(99, 40),
                  cnn_features=30,
                  rnn='LSTM',
                  multi_rnn=True,
                  attention=True,
                  dropout=0.2):
    '''
    Long-Short-Term-Memory model

    Parameters:\n
    input_shape (array): dimensions of the model input\n
    cnn_features (int): number of features for the first CNN Layer\n
    rnn (string [LSTM, GRU]): type of RNN to use in the model\n
    multi_rnn (bool): activate or deactivate the second RNN Layer\n
    attention (bool): activate or deactivate the Attention Layer\n
    dropout (int [0:1]): dropout level for Dense Layers

    Returns:\n
    tf.keras.Model: Model built with keras
    '''

    # Fetch input
    inputs = tf.keras.Input(shape=input_shape)
    reshape = tf.keras.layers.Reshape(
        input_shape=input_shape, target_shape=(int(input_shape[0]), int(input_shape[1]), 1))(inputs)

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
    if rnn not in ['LSTM', 'GRU']:
        raise ValueError(
            'rnn should be equal to LSTM or GRU. No model generated...')

    if rnn == 'LSTM':
        layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            60, return_sequences=True, dropout=dropout))(layer_out)
        if multi_rnn:
            layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                60, return_sequences=True, dropout=dropout))(layer_out)

    # GRU Layer
    if rnn == 'GRU':
        layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
            60, return_sequences=True, dropout=dropout))(layer_out)
        if multi_rnn:
            layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                60, return_sequences=True, dropout=dropout))(layer_out)

    # Attention Layer
    if attention:
        query, value = tf.keras.layers.Lambda(
            lambda x: tf.split(x, num_or_size_splits=2, axis=2))(layer_out)
        layer_out = tf.keras.layers.Attention(name='Attention')([query, value])

    # Classification Layer
    outputs = tf.keras.layers.Flatten()(layer_out)
    outputs = tf.keras.layers.Dense(512, activation='relu')(outputs)
    outputs = tf.keras.layers.Dense(30, activation='softmax')(outputs)

    # Output Model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model
