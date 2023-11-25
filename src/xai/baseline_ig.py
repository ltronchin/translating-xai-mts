import tensorflow as tf
import numpy as np
from utils import util_data
import torch
from tqdm import tqdm

class BaselineIntegratedGradientsCNN:
    def __init__(self, dataset, model_cnn, confidence_thr=0.95, compute_mean = False):
        """
        * For a signal predicted as crash event, we exploit as baseline the zero acceleration signal and the constant speed equal to  0.1 in a normalised scale
        * For a signal predicted as a non-crash event, we exploit as baseline the average of a subset of validation set signals predicted as a crash with higher confidence
        Args:
            dataset:
            model_cnn:
            compute_mean:
        """
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=1)

        X_acc_list = []
        X_pos_list = []
        y_pred_cnn_list = []
        for X_acc, X_pos, y, _ in tqdm(dataloader):

            label = y.detach().cpu().numpy()[0]
            if label == 0: # non-crash samples -> skip.
                continue

            X_acc = tf.convert_to_tensor(X_acc, dtype=tf.float32)
            X_pos = tf.convert_to_tensor(X_pos, dtype=tf.float32)

            y_pred_cnn = model_cnn.predict([X_acc, X_pos])
            if y_pred_cnn[0][0] >= confidence_thr:
                X_acc_list.append(X_acc.numpy()[0])
                X_pos_list.append(X_pos.numpy()[0])
                y_pred_cnn_list.append(y_pred_cnn[0][0])

        if compute_mean:
            self.top_reliable_acc = tf.reduce_mean(tf.convert_to_tensor(X_acc_list, tf.dtypes.float32), axis=0)[tf.newaxis, :, :]
            self.top_reliable_vel = tf.reduce_mean(tf.convert_to_tensor(X_pos_list, tf.dtypes.float32), axis=0)[tf.newaxis, :]
        else:  # select only the highest one
            y_pred_cnn_list = np.array(y_pred_cnn_list)
            idx = np.argmax(y_pred_cnn_list)
            self.top_reliable_acc = tf.convert_to_tensor(X_acc_list[idx], tf.dtypes.float32)[tf.newaxis, :, :]
            self.top_reliable_vel = tf.convert_to_tensor(X_pos_list[idx], tf.dtypes.float32)[tf.newaxis, :]

    def choose_baseline(self, i=0, input=tf.zeros([1, 2490, 3]), const_value=0.01, min_random=-0.1, max_random=0.1):
        """Select the baseline to compute Integrated Gradients
        Args
            i (Int): index of the switcher in order to select the baseline generation method.
            input (Tensor): A 3D or 2D tensor of float with shape (1, 2490, 3) or (1, 41) representing the acceleration signal or
                speed signal.
            class_to_explain: class of which we want an explanation;
            const_value (Float): value to set the baseline tensor if constant value is selected
            min_random (Float): inferior limit of uniform distribution from which to sample the values to set the
                baseline tensor if random is selected.
            max_random (Float)  superior limit of uniform distribution from which to sample the values to set the
                baseline tensor if random is selected.
            index (Int): index of the switcher.
        Returns:
            baseline (Tensor): A 3D or 2D tensor of floats with shape (1, 2490, 3) or (1, 41) representing the baseline for
                acceleration signal or speed signal
        """

        switcher = {
            0: tf.constant(const_value, shape=input.shape, dtype=tf.dtypes.float32), # Constant values
            1: tf.random.uniform(shape=input.shape, minval=min_random, maxval=max_random, dtype=tf.dtypes.float32), # Random values
            2: [self.top_reliable_acc,  self.top_reliable_vel]
            }
        baseline = switcher.get(i, 'Invalid baseline')

        return baseline
