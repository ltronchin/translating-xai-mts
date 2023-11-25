import copy
from functools import partial

import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
import tensorflow as tf

from . import lime_base
from . import scikit_image


class ImageExplanation(object):
    def __init__(self, image, image_scaled, segments):
        """Initialization function.

        Args:
            image: A numpy array representing the image.
            image_scaled: A numpy array with rescaled pixel values of the image.
            segments: A 2D numpy array that comes from skimage.segmentation.
        """
        self.image = image
        self.image_scaled = image_scaled
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}

    def get_image_and_mask(self, positive_only=True, negative_only=False, hide_rest=False,
                           num_features=5, min_weight=0.):
        """Get the image and its mask.

        Args:
            positive_only: If True, includes only superpixels that contribute positively to the prediction of the label.
            negative_only: If True, includes only superpixels that contribute negatively to the prediction of the label.
                           If both positive_only and negative_only are False, then both negative and positive contributions will be included.
                           Both cannot be True at the same time.
            hide_rest: If True, the non-explanation part of the image is made gray.
            num_features: The number of superpixels to include in the explanation.
            min_weight: The minimum weight of the superpixels to include in the explanation.

        Returns:
            Tuple of (image, mask, activation).
            Image is a 3D numpy array and mask is a 2D numpy array that can be used with skimage.segmentation.mark_boundaries.
        """
        # Assigns a label based on the class to explain.
        if self.class_to_explain[:] == 'crash':
            label = 0
        else:
            label = 1

        if positive_only & negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")
        segments = self.segments
        image = self.image.copy()
        exp = self.local_exp[label]
        mask = np.zeros(segments.shape, segments.dtype)
        activation = np.zeros(segments.shape)

        if hide_rest:
            # If hide_rest is True, temp only shows the regions/features that contribute positively or negatively.
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()

        # Extracts all the parameters of the regressor.
        all_fs = [x[0] for x in exp][:]
        regressor_parameters = [x[1] for x in exp][:]
        for i in range(len(regressor_parameters)):
            activation[segments == all_fs[i]] = regressor_parameters[i]

        if positive_only:
            fs = [x[0] for x in exp if x[1] > 0 and x[1] > min_weight][:num_features]
        if negative_only:
            fs = [x[0] for x in exp if x[1] < 0 and abs(x[1]) > min_weight][:num_features]

        if positive_only or negative_only:
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask, activation
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                # In the mask, regions that contribute positively are set to 1, those that contribute negatively are set to 0.
                mask[segments == f] = -1 if w < 0 else 1
                temp[segments == f] = image[segments == f].copy()
            return temp, mask

class LimeImageExplainer(object):
    """Explains predictions on time series data that are treated like images."""

    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None):
        """Initialization function.

        Args:
            kernel_width: The width of the kernel for the exponential kernel. Default is 0.25.
            kernel: Similarity kernel function that outputs weights in (0,1) range given euclidean distances and kernel width.
                    If None, an exponential kernel is used by default.
            verbose: If True, local prediction values from the linear model are printed.
            feature_selection: The method of feature selection. It can be 'forward_selection', 'lasso_path', 'none' or 'auto'.
            random_state: An integer or numpy.RandomState that will be used to generate random numbers.
                          If None, the random state will be initialized with an internal numpy seed.
        """

        # Kernel width is converted to a float.
        kernel_width = float(kernel_width)

        # If no kernel is provided, the default kernel is an exponential one.
        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        # Here, a new kernel function is defined using partial application. This function takes the distance as an input and the kernel width is predefined.
        kernel_fn = partial(kernel, kernel_width=kernel_width)

        # The random state is checked and set.
        self.random_state = check_random_state(random_state)

        # Feature selection method is set.
        self.feature_selection = feature_selection

        # LimeBase is initialized with the kernel function, verbosity, and the random state. This will be used to generate explanations.
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)

    def explain_instance(self,
                         image,
                         classifier_fn,
                         class_to_explain,
                         labels=(1,),
                         hide_color=None,
                         top_labels=5,
                         num_features=100000,
                         num_samples=1000,
                         batch_size=10,
                         distance_metric='cosine',
                         model_regressor=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            image: time series signals considered like an image
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            class_to_explain: class to be explained.
            hide_color: set the pixel values of the perturbed regions
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: batch of samplex feed to the pre-trained network
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()

        Returns:
            An ImageExplanation object (see lime_image_custom.py) with the corresponding
            explanations.
        """
        # image: list of two elements containing the acceleration and velocity vectors ([acceleration, velocity])
        # Saving the arrays contained in the list in two distinct variables image_acc and image_vel
        image_acc = tf.make_ndarray(tf.make_tensor_proto(image[0]))
        image_vel = tf.make_ndarray(tf.make_tensor_proto(image[1]))
        image = image_acc  # Explanation on acceleration

        segmentation_fn = scikit_image.SegmentationAlgorithm('felzenszwalb', scale=1, sigma=0.8,  min_size=20, multichannel=False)
        # The "pixel" values of the image that represent the accelerations are of float type and have
        # excursions of both negative and positive values. The segmentation algorithm used, chosen because
        # it allows working on grayscale images, assumes that for float data the range of values is
        # between -1 and 1 or between 0 and 1. Since accelerations can also be negative, the
        # sample is normalized so that the acceleration values are between -1 and 1.
        image_scaled = 2 * ((image - np.amin(image)) / (np.amax(image) - np.amin(image))) - 1
        try:
            # segments: segmentation of the image, defines regions in the starting image
            segments = segmentation_fn(image_scaled)
        except ValueError as e:
            raise e

        fudged_image = image.copy()
        # hide_color: value with which to overwrite the pixels of the regions identified by segmentation when
        # generating perturbed samples.
        # If None, calculate the average pixel per region and instead of setting them to hide_color it sets them to the average value
        if hide_color is None:
            # Loop over all identified regions
            for x in np.unique(segments):
                region_mean_value = np.mean(image[segments == x][:])
                fudged_image[segments == x] = (region_mean_value)
        else:
            fudged_image[:] = hide_color

        # Creating an instance of the ImageExplanation class
        ret_exp = ImageExplanation(image, image_scaled, segments)

        # Selecting the label to be explained
        ret_exp.class_to_explain = class_to_explain
        top = labels

        data, labels = self.data_labels(image,
                                        image_vel,
                                        fudged_image,
                                        segments,
                                        classifier_fn,
                                        num_samples,
                                        batch_size=batch_size)

        # Computing the distance between the perturbed samples and the original sample
        original_data_point = data[0].reshape(1, -1)
        distances = sklearn.metrics.pairwise_distances(data, original_data_point, metric=distance_metric).ravel()

        # Creating a label vector where the first column is the probability that the sample is a crash and the second
        # contains the complementary values
        labels = np.c_[labels, 1 - labels]

        if top_labels:
            # Creating a vector of indices sorted from the class (column) with the least likelihood of being predicted to the one with
            # the most likelihood of being predicted for the unperturbed sample (hence labels[0]).
            order_label_index = np.argsort(labels[0], axis=-1)
            # Selecting the top_labels predictions, considering the last elements of the vector (order from smallest
            # to largest along the columns)
            top = order_label_index[-top_labels:]
            ret_exp.top_labels = list(top)
            # Reversing the elements: in this way the vector will be ordered from the class with the most likelihood
            # of being predicted to the one with the least likelihood
            ret_exp.top_labels.reverse()

        # Looping through all the labels (columns) for which we want to obtain an explanation. Top contains the columns for
        # which we want to obtain an explanation. If there is only one neuron at the output (sigmoid), top = 0.
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(data,
                                                                               labels,
                                                                               distances,
                                                                               label,
                                                                               num_features,
                                                                               model_regressor=model_regressor,
                                                                               feature_selection=self.feature_selection)
        return ret_exp

    def data_labels(self,
                    image,
                    image_vel,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: numpy array, the image
            image_vel: numpy array, the speed data associated with the image
            fudged_image: numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        # The number of features equals the number of different REGIONS identified by the segmentation algorithm.
        # np.unique creates a vector containing all the different values identified in segments.
        n_features = np.unique(segments).shape[0]

        # data -- matrix of size (num_samples, n_features) filled with 0s and 1s randomly generated.
        #   num_samples: represents the number of different perturbations to be made on the image, thus it regulates
        #   the number of different samples that will be used to train the linear classifier (for each
        #   perturbation a new sample is obtained).
        #   n_features: number of regions identified in the segmentation.
        # Each row of this matrix is used to perturb the input; where there's a 0, the corresponding area of the
        # original image is "turned off" (made zero).
        data = self.random_state.randint(0, 2, num_samples * n_features).reshape((num_samples, n_features))

        # List for storing the labels that will be associated by the network to the perturbed samples
        labels = []
        data[0, :] = 1  # The first row does not perturb the image (row composed of all 1s)
        imgs = []

        # Select a row of data
        for row in data:
            # Save a copy of the original image in temp
            temp = copy.deepcopy(image)

            # Initialize a mask of False the size of segments
            mask = np.zeros(segments.shape).astype(bool)

            # Retrieve the column indices where there are 0s
            zeros = np.where(row == 0)[0]

            # For each zero in zeros, set the corresponding region in mask to True
            for z in zeros:
                mask[segments == z] = True

            # Apply the mask to the original image
            temp[mask] = fudged_image[mask]
            imgs.append(temp)

            if len(imgs) == batch_size:
                imgs = np.array(imgs)

                # To make a prediction with the network, you need to pass in a list of 2 elements containing acceleration and speed.
                # Repeat the speed vector batch_size times to generate a "batch_size x 41 x 1" vector.
                # np.tile repeats the imgs_vel vector batch_size times
                imgs_vel = np.tile(image_vel, (batch_size, 1))
                preds = classifier_fn([imgs, imgs_vel])

                labels.extend(preds)
                imgs = []

        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)

        return data, np.array(labels)
