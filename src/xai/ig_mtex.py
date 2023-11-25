import tensorflow as tf
import sys

class IntegratedGradients:
    def __init__(self, model_cnn, m_steps = 50):
        """
        Args:
          model_cnn (keras.Model): A trained model to generate predictions and inspect.
          m_steps(int): integer corresponding to the number of linear interpolation steps for computing an approximate integral.
        """
        self.model_cnn = model_cnn
        self.m_steps = m_steps  # number of interpolated images to generate

    def interpolate_input(self, baseline, input, alphas):
        """Generate m_steps interpolated inputs along a linear path at alpha intervals between baseline and input features.
        Args:
          baseline (Tensor): A 3D or 2D tensor of floats with the shape (1, 2490, 3) or (1, 41)
          input (Tensor): A 3D or 2D tensor of floats with the shape (1, 2490, 3) or (1, 41)
          alphas (Tensor): A 1D tensor of uniformly spaced floats with the shape (m_steps,) corresponding to the
            constant for varying the intensity of the interpolated images between the baseline and input.
        Returns:
          interpol_images (Tensor): A 3D or 2D tensor of floats with the shape (m_steps, 2490, 3) or (m_steps, 41)
            corresponding to the small steps in the feature space between the baseline and the input
        """

        for i in range(input.ndim - 1):
            alphas = tf.expand_dims(alphas, axis = -1)
        delta = input - baseline
        interpol_images = baseline + alphas * delta
        return interpol_images

    def compute_gradients(self, interpolated, idx_to_explain):
        """Compute gradients of model predicted probabilities with respect to inputs.
        Args:
          interpolated (Tensor): A two elements list.
            interpolated[0]: A 3D tensor of floats with the shape (m_steps, 2490, 1) corresponding to the interpolated path of acceleration
            interpolated[1]: A 2D tensor of floats with the shape (m_steps, 41), corresponding to the interpolated path of speeds;
          idx_to_explain (int): integer corresponding to the index of the class to explain.
        Returns:
          gradients(Tensor): A two elements list:
            gradients[0]: A 3D tensor of floats with the  shape (m_steps, 2490, 3) corresponding to the gradient of output respect acceleration input
            gradients[1]: A 2D tensor of floats with the  shape (m_steps, 41) corresponding to the gradient of output respect speed input
        """
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.model_cnn(interpolated)[:]
            pred = pred[:, idx_to_explain]

            return tape.gradient(pred, interpolated)

    def integral_approximation(self, gradients):
        """Compute numerical approximation of integral from gradients.
            Args:
              gradients (Tensor): A 3D or 2D tensor of floats with the shape (m_steps, 2490, 3) or (m_steps, 41)
            Returns:
              integrated_gradients(Tensor): A 2D or 1D tensor of floats with the shape (2490, 3) or (41,).
        """
        integrated_gradients = tf.math.reduce_mean(gradients, axis=0)  # sum the m gradients and divide by m steps
        return integrated_gradients

    def convergence_check(self, baseline, x, IG_attributions_acc, class_to_explain):
        """Check to pick correct number of steps for Integrated Gradients approximations using the completeness axiom
        Args:
            baseline (Tensor): A 3D image tensor with the shape (1, 2490, 3) with the same shape as the acc tensor
            x (Tensor): A 3D acceleration tensor with the shape (1, 2490, 3).
            IG_attributions_acc (Tensor): A 2D tensor with the shape (2490, 3)
            class_to_explain (Str): class to explain
        Returns:
             delta (float64): difference between the model prediction at the input and the model prediction at the baseline
        """

        baseline_score =  self.model_cnn.predict(baseline)[0][0].astype('float64')
        input_score = self.model_cnn.predict(x)[0][0].astype('float64')

        ig_score_acc = tf.math.reduce_sum(IG_attributions_acc)

        convergence = (ig_score_acc - (input_score - baseline_score))

        try:
            tf.debugging.assert_near(ig_score_acc, (input_score - baseline_score), rtol=0.05)
            convergence = 1
            #tf.print('Approximation accuracy within 5%.', output_stream=sys.stdout)
        except tf.errors.InvalidArgumentError:
            tf.print('IG warning: Increase or decrease m_steps to increase approximation accuracy.\n', output_stream=sys.stdout)
            print('Baseline score: {}'.format(baseline_score))
            print('Input score: {}'.format(input_score))
            print('IG score: {}'.format(ig_score_acc))
            print('Convergence delta: {}'.format(convergence))
            convergence = convergence

        delta = input_score - baseline_score

        return delta, convergence

    def integrated_gradients(self, x, baseline, idx_to_explain, normalize=True):
        """
        Args:
          x (Tensor): A 3D acceleration tensor with the shape (1, 2490, 3).
          baseline (Tensor): A 3D image tensor with the shape (1, 2490, 3) with the same shape as the acc tensor
          idx_to_explain (int): integer corresponding to the index of the class to explain.
          normalize (bool): If True, normalize the gradients with respect to the input
        Returns:
            integrated_gradients(Tensor): A 2D tensor of floats with the same shape as the input tensor (2490, 3)
                representing the IG importance map for acceleration
        """

        # Generation of alphas and baseline
        #   Generates a sequence of values that starts from 0.0 up to 1.0 with
        #   increment equal to stop - start / num - 1 (increment equal to 1/m_steps).
        self.alphas = tf.linspace(start = 0.0, stop = 1.0, num = self.m_steps + 1)

        # Generation of interpolated inputs between the baseline and the input
        interpolated_input= self.interpolate_input(baseline = baseline, input = x, alphas = self.alphas)

        # Calculation of the gradient between the model's output for the target class and
        # the interpolated inputs. Returns the list of gradients for acceleration
        # and speed (a list of two elements)
        path_gradients = self.compute_gradients(interpolated_input, idx_to_explain)

        # Integral approximation by calculating the average of the gradients
        IG_avg = self.integral_approximation(gradients = path_gradients)

        # Scale the IG with respect to the input
        ig = (x[0] - baseline[0]) * IG_avg

        delta, convergence = self.convergence_check(baseline, x, ig, idx_to_explain)

        # Normalization between 0 and 1
        if normalize:
            ig = (ig - tf.math.reduce_min(ig)) / (tf.math.reduce_max(ig) - tf.math.reduce_min(ig))

        return tf.make_ndarray(tf.make_tensor_proto(ig)), delta, convergence

