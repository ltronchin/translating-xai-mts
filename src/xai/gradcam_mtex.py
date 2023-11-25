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
        self.last_conv_layer_name = 'conv4_1d_dropout'
        self.classifier_layer_names = ['flatten', 'fc1','output_layer']
        self.vel_input_names = []

    def make_models(self):
        """
        Function to create two models, given the convolutional layer after which to "break" the network.
        This operation is necessary to retrieve the activations of the last convolutional layer.

        Args:
            None
        Returns:
            None
        """
        self.grad_model = keras.models.Model(
            [self.model_cnn.inputs], [self.model_cnn.get_layer(self.last_conv_layer_name).output, self.model_cnn.output]
        )

    def compute_gradient(self, img_array, idx_to_explain):
        """
        Function to compute gradient of the predicted class with respect to the feature maps of the last convolutional layer.

        Args:
            img_array (Tensor): The acceleration signal (considered as the "image" to be explained)
            idx_to_explain (Int): The class to explain

        Returns:
            grads (Tensor): Gradient of the predicted class with respect to the feature maps of the last convolutional layer
            last_conv_layer_output (Tensor): Output of the last convolutional layer
            preds (Tensor): The predictions of the model
        """

        with tf.GradientTape() as tape:
            conv_layer_output, preds = self.grad_model(img_array)
            pred_index = tf.argmax(preds[0])
            pred_class = preds[:, idx_to_explain] # pred_class = preds[:, pred_index]

        grads = tape.gradient(pred_class, conv_layer_output)

        return grads, conv_layer_output, preds

    def run_gradcam(self, img_like, repeats, relu, normalize, idx_to_explain):
        """
        Compute the Grad-CAM heatmap

        Args:
            img_like (Tensor): A tensor of floats corresponding to the MTS sample
            repeats (Int):
            relu (Bool): Whether or not to apply the ReLU function to the heatmap
            normalize (Bool): Whether or not to normalize the heatmap
            idx_to_explain (Int): The class to explain
        Returns:
            heatmap (Array): The Grad-CAM heatmap
            preds (Float): The probability of the 'crash' class

        """
        # Compute gradient
        grads, last_conv_layer_output, preds = self.compute_gradient(img_like, idx_to_explain)
        # Compute the importance weight vector for each feature map
        importance_weight = self.compute_importance_weight(grads)
        # Compute the heatmap by multiplying the importance for each feature map
        heatmap = self.compute_gradcam_heatmap(importance_weight, last_conv_layer_output, img_like, repeats=repeats, relu=relu, normalize=normalize)

        return heatmap, preds

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

    def compute_gradcam_heatmap(self, pooled_grads, last_conv_layer_output, img_like, repeats=False, relu=False, normalize=True):
        """
        Compute the Grad-CAM heatmap by multiplying the importance for each feature map.

        Args:
            pooled_grads (Tensor): Importance weights for each feature map
            last_conv_layer_output (Tensor): Output of the last convolutional layer
            img_like (Tensor): A tensor of floats corresponding to the MTS sample
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
            heatmap = np.repeat(heatmap, repeats=(img_like.shape[1] / heatmap.shape[0]))
            # Pad
            if heatmap.shape[0] < img_like.shape[1]:
                heatmap = np.pad(heatmap, (0, img_like.shape[1] - heatmap.shape[0]), mode='edge')
        return heatmap

    def perturb_time_series(self, x, axis_to_explain):
        """
        Perturb the time series.

        Args:
            x (Tensor): A 3D tensor of floats corresponding to the acceleration sample
            axis_to_explain (Int): The axis of the acceleration data to perturb

        Returns:
            Tensor: The perturbed time series
        """
        x = x[0].numpy()
        zero = np.zeros(x.shape)
        zero[:, axis_to_explain] = x[:, axis_to_explain]

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

