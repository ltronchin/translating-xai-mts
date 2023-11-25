import numpy as np

def select_class_to_explain(i, acc, vel, label):
    """Function to define which class to explain:
    0: explains the "crash" class regardless of the prediction from the CNN or the ground truth
    1: explains the "non_crash" class regardless of the prediction from the CNN or the ground truth
    2: explains the class according to the truth ("crash" if the sample is truly a "crash" and vice versa)
    3: explains the class according to the CNN prediction
    Args:
        i (Int): index of the switcher.
        acc (Tensor): A 3D tensor of floats with the shape (1, 2490, 1) corresponding to the acceleration sample
        vel (Tensor): A 3D tensor of floats with the shape (1, 41, 1) corresponding to the speed sample
        label (Int): ground truth class.
    Returns:
        class_to_explain (String)
    """

    def cnn_prediction():
        thr_cnn = 0.5061745

        # Check if the tensor are Pytorch and convert to Tensorflow
        if isinstance(acc, torch.Tensor):
            acc = acc.detach().numpy()
        if isinstance(vel, torch.Tensor):
            vel = vel.detach().numpy()

        pred = model_cnn.predict([acc, vel])
        if np.uint8(pred > thr_cnn)[0] == 1:
            return 'crash'
        else:
            return 'non_crash'

    def ground_truth():
        if label == 1:
            return 'crash'
        else:
            return 'non_crash'

    switcher = {
        0: lambda: 'crash',
        1: lambda: 'non_crash',
        2: ground_truth,
        3: cnn_prediction,
    }
    func = switcher.get(i, lambda: 'Invalid')

    return func()
