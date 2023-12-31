from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, Concatenate, MaxPool1D


class CNNModel():
    def __init__(self, weight_filepath):
        self.weight_filepath = weight_filepath

    def load_cnn_model(self):
        model_name = "cnn_1d_001"
        samples_acc = 2490
        samples_pos = 41

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

        model = keras.Model([acc_input, pos_input], out, name=model_name)

        model.load_weights(self.weight_filepath)  # Caricamento dei pesi della rete

        return model