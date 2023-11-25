import os
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, Concatenate, MaxPool1D, LSTM, Conv2D, Reshape, Dropout, Activation
from tensorflow.keras import regularizers

def get_model(model_name):
    if 'cnn' in model_name:
        return build_model_cnn
    elif 'mtex' in model_name:
        return build_model_mtex
    else:
        raise ValueError("Model name not valid")

def build_model_cnn(model_dir, model_name = "cnn_1d_001", samples_acc=2490, samples_pos=41):

    acc_input = Input((samples_acc, 3), name="input_acc")
    pos_input = Input((samples_pos,), name="input_pos")

    conv1 = Conv1D(16, kernel_size=5, strides=1, activation='relu', padding="causal", name="conv1")(acc_input)
    conv1 = MaxPool1D(2, padding="valid", name="pool1")(conv1)
    conv2 = Conv1D(32, kernel_size=5, strides=1, activation='relu', padding="causal", name="conv2")(conv1)
    conv2 = MaxPool1D(2, padding="valid", name="pool2")(conv2)
    fc = Flatten(name="flatten")(conv2)
    l1_pos = Dense(100, activation="relu", name="fc1_vel")(pos_input)
    fc0 = Concatenate(name="concatenate")([fc, l1_pos])
    fc1 = Dense(500, activation="relu", name="fc1")(fc0)
    fc2 = Dense(250, activation="relu", name="fc2")(fc1)
    fc3 = Dense(100, activation="relu", name="fc3")(fc2)
    out = Dense(1, activation="sigmoid", name="output_layer")(fc3)

    model = Model([acc_input, pos_input], out, name=model_name)

    model.load_weights(os.path.join(model_dir, f'{model_name}.hdf5'))

    return model

def build_model_mtex(model_dir, model_name, input_shape, n_class):

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

    z = Flatten(name='flatten')(y)
    z = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.2), name='fc1')(z)
    output_layer = Dense(n_class, activation="softmax", name='output_layer')(z)

    model = Model(input_layer, output_layer)

    model.load_weights(os.path.join(model_dir, f'{model_name}.hdf5'))

    return model

def predict_model(model, data_loader):
    pass
