from tensorflow import keras
import tensorflow as tf
import numpy as np
class GradCAM:
    def __init__(self, model_cnn):
        """
        Initialize the GradCAM object.
        Args:
            model_cnn (Model): The CNN model for which the GradCAM will be computed.
        """
        self.model_cnn = model_cnn
        self.last_conv_layer_name = 'conv2'
        self.classifier_layer_names = ['pool2', 'flatten', 'concatenate',  'fc1',  'fc2',  'fc3', 'output_layer']
        self.vel_input_names = ['fc1_vel']

    def make_models(self):
        """
        Function to create two models, given the convolutional layer after which to "break" the network.
        This operation is necessary to retrieve the activations of the last convolutional layer.
    
        Args:
            None
    
        Returns:
            last_conv_layer_model (Model): A model that has the output of the last convolutional layer
            classifier_model (Model): A model that includes the layers after the last convolutional layer
        """

        # Model creation for speed
        x = self.model_cnn.input[1]
        for layer_name in self.vel_input_names:
            x = self.model_cnn.get_layer(layer_name)(x)
        vel_model = keras.Model(self.model_cnn.input[1], x)

        # Selection of the last convolutional layer of model_cnn (using the get_layer function of the Model class from keras)
        last_conv_layer = self.model_cnn.get_layer(self.last_conv_layer_name)
        # Creation of the model that includes all layers from the CNN input to the last convolutional layer
        self.last_conv_layer_model = keras.Model(self.model_cnn.inputs[0], last_conv_layer.output)

        classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer_name in self.classifier_layer_names:
            # For the 'concatenate' layer, it is necessary to combine the part of the model that takes speed as input
            # with the part of the model that takes the output from the last convolutional layer as input
            if layer_name[:] == 'concatenate':
                x = self.model_cnn.get_layer(layer_name)([x, vel_model.output])
            else:
                x = self.model_cnn.get_layer(layer_name)(x)

        self.classifier_model = keras.Model([classifier_input, self.model_cnn.input[1]], x)  # model creation

    def compute_gradient(self,
                         img_array,
                         vel,
                         crash='crash'):
        """
        Function to compute gradient of the predicted class with respect to the feature maps of the last convolutional layer.

        Args:
            img_array (Tensor): The acceleration signal (considered as the "image" to be explained)
            vel (Tensor): A 2D tensor of floats corresponding to the speed sample
            crash (String): Class to compute Grad-CAM for ('crash' or 'non_crash')

        Returns:
            grads (Tensor): Gradient of the predicted class with respect to the feature maps of the last convolutional layer
            last_conv_layer_output (Tensor): Output of the last convolutional layer
            preds (Tensor): The predictions of the model
        """
        with tf.GradientTape() as tape:
            last_conv_layer_output = self.last_conv_layer_model(
                img_array)  # Compute activations of the last conv layer and make the tape watch it
            tape.watch(last_conv_layer_output)
            preds = self.classifier_model([last_conv_layer_output, vel])
            preds_non_crash = 1 - preds

        if crash[:] == 'crash':
            grads = tape.gradient(preds, last_conv_layer_output)
        else:
            grads = tape.gradient(preds_non_crash, last_conv_layer_output)

        return grads, last_conv_layer_output, preds

    def run_gradcam(self,
                    acc,
                    vel,
                    repeats,
                    relu,
                    normalize,
                    class_to_explain):
        """
        Compute the Grad-CAM heatmap and the probability of the 'crash' class.

        Args:
            acc (Tensor): A 3D tensor of floats corresponding to the acceleration sample
            vel (Tensor): A 2D tensor of floats corresponding to the speed sample
            repeats (Int):
            relu (Bool): Whether or not to apply the ReLU function to the heatmap
            normalize (Bool): Whether or not to normalize the heatmap
            class_to_explain (String): The class to compute Grad-CAM for ('crash' or 'non_crash')

        Returns:
            heatmap (Array): The Grad-CAM heatmap
            crash_probability (Float): The probability of the 'crash' class

        """
        # Compute gradient
        grads, last_conv_layer_output, crash_probability = self.compute_gradient(
            acc,
            vel,
            class_to_explain
        )
        # Compute the importance weight vector for each feature map
        importance_weight = self.compute_importance_weight(grads)

        # Compute the heatmap by multiplying the importance for each feature map
        heatmap = self.compute_gradcam_heatmap(
            importance_weight, last_conv_layer_output, repeats=repeats, relu=relu, normalize=normalize
        )

        return heatmap, crash_probability

    def compute_importance_weight(self, grads):
        """
        Compute the importance weight for each feature map.
        Args:
            grads (Tensor): Gradient of the predicted class with respect to the feature maps of the last convolutional layer

        Returns:
            pooled_grads (Tensor): Importance weights for each feature map, tensor of size ([1, K]). K = 32 for the crash_alert system
        """
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        return pooled_grads

    def compute_gradcam_heatmap(self, pooled_grads, last_conv_layer_output, repeats=False, relu=False, normalize=True):
        """
        Compute the Grad-CAM heatmap by multiplying the importance for each feature map.

        Args:
            pooled_grads (Tensor): Importance weights for each feature map
            last_conv_layer_output (Tensor): Output of the last convolutional layer
            repeats (Int): The number of times each feature map is repeated when upsampling the heatmap
            relu (Bool): Whether or not to apply the ReLU function to the heatmap
            normalize (Bool): Whether or not to normalize the heatmap

        Returns:
            heatmap (Array): The Grad-CAM heatmap
        """
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):  # loop over the feature maps
            last_conv_layer_output[:, i] *= pooled_grads[i]

        heatmap = np.mean(last_conv_layer_output, axis=-1)

        if relu:
            heatmap = np.maximum(heatmap, 0)  # apply ReLu
        if normalize:
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # normalize between 0 and 1
        if repeats:
            heatmap = np.repeat(heatmap, repeats=2)

        return heatmap

    def perturb_time_series(self, acc, axis_to_explain):
        """
        Perturb the time series.

        Args:
            acc (Tensor): A 3D tensor of floats corresponding to the acceleration sample
            axis_to_explain (Int): The axis of the acceleration data to perturb

        Returns:
            Tensor: The perturbed time series
        """
        acc = acc[0].numpy()
        zero = np.zeros(acc.shape)
        zero[:, axis_to_explain] = acc[:, axis_to_explain]

        return tf.expand_dims(tf.convert_to_tensor(zero), axis=0)

    def ricombination_method(self, heatmap_orig, crash_p_orig, heatmap_pert, crash_p_pert):
        """
        Recombine the original and perturbed heatmaps.

        Args:
            heatmap_orig (Array): The original Grad-CAM heatmap
            crash_p_orig (Float): The probability of the 'crash' class for the original sample
            heatmap_pert (Array): The heatmap of the perturbed sample
            crash_p_pert (Float): The probability of the 'crash' class for the perturbed sample

        Returns:
            Array: The recombined heatmap
        """
        heatmap_perturbed_axis = heatmap_orig * (
            np.abs(crash_p_orig - crash_p_pert)[0][0]) + heatmap_pert * (1 - np.abs(crash_p_orig - crash_p_pert))[0][0]
        return heatmap_perturbed_axis

