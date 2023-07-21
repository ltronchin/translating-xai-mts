import numpy as np
import os
from scipy.io import savemat, loadmat
import tensorflow as tf

class UploadData():
    def __init__(self, path):
        self.path = path

    def load(self, dataset_name, convert_to_tensor, subsampling = False):
        ''' Function to load the data of multivariate time series
        :param dataset_name (str): name of dataset split to load ('test', 'valid', 'train')
        :param convert_to_tensor (bool): boolean condition to convert numpy data in tensor of float32
        :return:
        '''
        if subsampling:
            with open(os.path.join(self.path + "/y_" + dataset_name + "_subsampling.npy"), 'rb') as f:
                self.label = np.load(f)
            with open(os.path.join(self.path + "/X_pos_" + dataset_name + "_scaled_subsampling.npy"), 'rb') as f:
                self.pos = np.load(f)
            with open(os.path.join(self.path + "/X_acc_" + dataset_name + "_scaled_subsampling.npy"), 'rb') as f:
                self.acc = np.load(f)
        else:
            with open(os.path.join(self.path + "/y_" + dataset_name + ".npy"), 'rb') as f:
                self.label = np.load(f)
            with open(os.path.join(self.path + "/X_pos_" + dataset_name + "_scaled.npy"), 'rb') as f:
                self.pos = np.load(f)
            with open(os.path.join(self.path + "/X_acc_" + dataset_name + "_scaled.npy"), 'rb') as f:
                self.acc = np.load(f)

        print(dataset_name)
        print("Accelerazioni: {}".format(self.acc.shape))
        print("Posizioni: {}".format(self.pos.shape))
        print("Label: {}".format(self.label.shape))

        if convert_to_tensor:
            self.pos = tf.convert_to_tensor(self.pos, tf.dtypes.float32)
            self.acc = tf.convert_to_tensor(self.acc, tf.dtypes.float32)

        return self.acc, self.pos, self.label

    def load_matlab_data(self, xai_method, file_name, dataset_name):
        data_crash = loadmat(os.path.join(self.path + "/explanation_crash_row/" + xai_method + "/" + dataset_name + "/" + file_name + "_" + dataset_name + '.mat'))[file_name]
        data_non_crash = loadmat(os.path.join(self.path + "/explanation_non_crash_row/" + xai_method + "/" + dataset_name + "/" + file_name + "_" + dataset_name + '.mat'))[file_name]

        return data_crash, data_non_crash





