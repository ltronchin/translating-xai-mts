from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Conv1D, Conv2D, Flatten, Concatenate, MaxPool1D
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Input, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def mtex_cnn(input_shape, n_class):
    """
    MTEX-CNN model


    Parameters
    ----------
    input_shape: array
        Input shape array

    n_class: integer
        Number of classes


    Returns
    -------
    model: model
        MTEX-CNN Model
    """
    n = input_shape[0]
    k = input_shape[1]
    input_layer = Input(shape=(n, k, 1))

    a = Conv2D(filters=64, kernel_size=(8, 1), strides=(2, 1),  padding="same", input_shape=(n, k, 1), name="conv1")(input_layer)
    a = Activation("relu", name="conv1_activation")(a)
    a = Dropout(0.4, name='conv1_dropout')(a)

    a = Conv2D(filters=128, kernel_size=(6, 1), strides=(2, 1), padding="same", name="conv2")(a)
    a = Activation("relu", name="conv2_activation")(a)
    a = Dropout(0.4, name='conv2_dropout')(a)

    a = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), name="conv3_reduced")(a)
    a = Activation("relu", name="conv3_reduced_activation")(a)
    x = Reshape((int(n / 4), k))(a)

    b = Conv1D(filters=128, kernel_size=4, strides=2, input_shape=(int(n / 4), k), name="conv4_1d")(x)
    b = Activation("relu", name="conv4_1d_activation")(b)
    y = Dropout(0.4, name="conv4_1d_dropout")(b)

    z = Flatten()(y)
    z = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.2), name='fc1')(z)
    output_layer = Dense(n_class, activation="softmax", name='output_layer')(z)

    model = Model(input_layer, output_layer)

    print("MTEX-CNN Model Loaded")
    return model