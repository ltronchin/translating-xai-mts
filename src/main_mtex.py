import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./", "./src/xai"])


import tensorflow as tf
import torch
import numpy as np
import pandas as pd

import datetime
from scipy.io import savemat
import psutil
import yaml

# Custom imports
from src.utils import util_report
from src.utils import util_general
from src.utils import util_data
from src.utils import util_model
from src.utils import util_xai

from xai import gradcam_mtex, lime_time_series_mtex, ig_mtex, baseline_ig

if __name__ == '__main__':

    print("Upload configuration file")
    with open('./configs/xai_basicmotions.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Seed everything
    util_general.seed_all(cfg['seed'])

    # Parameters
    print("Parameters")
    worker = cfg['device']['worker']
    model_name = cfg['model']['model_name']
    dataset_name = cfg['data']['dataset']
    modes = cfg['data']['modes'] #  list(cfg['data']['modes'].keys())


    steps = ['train', 'valid', 'test']
    classes = cfg['data']['classes']
    class_to_explain = cfg['xai']['class_to_explain']

    # Device
    device = torch.device(worker if torch.cuda.is_available() else "cpu")
    num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
    print(f'device: {device}')

    # Files and Directories
    print("Files and directories")
    fold_dir = os.path.join(cfg['data']['fold_dir'], dataset_name)
    data_dir = os.path.join(cfg['data']['data_dir'], dataset_name)

    # Model dir
    model_dir = os.path.join(cfg['model']['model_dir'], dataset_name)
    util_general.create_dir(model_dir)

    # Report dir
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    reports_dir = os.path.join(cfg['reports']['reports_dir'], dataset_name, f"xai_{class_to_explain}_{time}")  # folder to save results
    util_general.create_dir(reports_dir)

    # Data
    (X_train, y_train, X_test, y_test, y_train_nonencoded, y_test_nonencoded) = util_data.import_data(data_dir=data_dir, dataset_name=dataset_name, class_to_idx=classes, randomize=True)
    data_iterator = iter(X_test)
    label_iterator = iter(y_test)

    # Upload pretrained models (tensorflow)
    model = util_model.build_model_mtex(model_dir=model_dir, model_name=model_name, input_shape=X_train.shape[1:], n_class=y_train.shape[1])

    # Iterate over the dataset
    counter = 0
    while True:
        if counter > cfg['xai']['n_samples_to_explain']:
            break

        x = next(data_iterator)
        y = next(label_iterator)

        # Explain the class predicted by the model
        if class_to_explain is None:
            idx_to_explain = np.argmax(y)
        else:
            idx_to_explain = classes[class_to_explain]

        n = x.shape[0]
        k = x.shape[1]

        reports_dir_current = os.path.join(reports_dir, f'idx_exp_{idx_to_explain}-{counter:05d}')
        util_general.create_dir(reports_dir_current)

        if cfg['reports']['snap']:
            util_report.plot_mts(outdir=reports_dir_current, x=x, info=f'mts')

        # To Tensoflow (the network was deployed from the company Tensorflow)
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        # Add dimension
        x = tf.expand_dims(x, axis=0)

        # Gradcam
        if 'gradcam' in cfg['xai']['methods']:
            reports_dir_gradcam =  os.path.join(reports_dir_current, 'gradcam')
            os.makedirs(reports_dir_gradcam)

            gCAM = gradcam_mtex.GradCAM(model)
            gCAM.make_models()

            heatmap_original, p_orig = gCAM.run_gradcam(
                img_like=x,
                repeats=True,
                relu=False,
                normalize=False,
                idx_to_explain=idx_to_explain
            )

            heatmap_perturbed_tilde = np.ones(shape=(x[0].shape[0], x[0].shape[1]))
            gradcam_heatmap = np.ones(shape=(x[0].shape[0], x[0].shape[1]))
            for ax in np.arange(k):

                heatmap_name = f'var_{ax}'
                # Perturb multivariate series.
                x_gradcam = gCAM.perturb_time_series(x, axis_to_explain=ax)

                # Run GradCAM on perturbed multivariate series.
                heatmap_perturbed_tilde[:, ax], p_pert = gCAM.run_gradcam(
                    img_like=x_gradcam,
                    repeats=True,
                    relu=False,
                    normalize=True,
                    idx_to_explain=idx_to_explain
                )

                gradcam_heatmap[:, ax] = gCAM.ricombination_method(
                    heatmap_original,
                    p_orig,
                    heatmap_perturbed_tilde[:, ax],
                    p_pert
                )

                if cfg['reports']['snap']:
                    util_report.plot_mts(outdir=reports_dir_gradcam, x=x_gradcam, info=f'mts_{heatmap_name}')
                    util_report.plot_mts_heatmap(outdir=reports_dir_gradcam, x=x_gradcam, heatmap=heatmap_perturbed_tilde[:, ax], info=f'mts_heatmap_{heatmap_name}')

            util_report.plot_heatmap(outdir=reports_dir_gradcam, x=x, heatmap=gradcam_heatmap, info=f'gradcam')

        # Integrated gradients
        if 'ig' in cfg['xai']['methods']:
            reports_dir_ig = os.path.join(reports_dir_current, 'ig')
            os.makedirs(reports_dir_ig)

            baseline = tf.constant(0.0, shape=x.shape, dtype=tf.dtypes.float32)  # Constant values
            ig_method = ig_mtex.IntegratedGradients(model, m_steps=cfg['xai']['ig']['m_steps'])

            if cfg['reports']['snap']:
                util_report.plot_mts(outdir=reports_dir_ig, x=baseline, info=f'baseline')

            ig_heatmap, delta, convergence = ig_method.integrated_gradients(x, baseline, idx_to_explain, normalize=True)

            util_report.plot_heatmap(outdir=reports_dir_ig, x=x, heatmap=ig_heatmap,  info=f'ig')
            print('network activation using the signal and the baseline: ', delta)

        # LIME
        if 'lime' in cfg['xai']['methods']:
            reports_dir_lime = os.path.join(reports_dir_current, 'lime')
            os.makedirs(reports_dir_lime)
            explainer = lime_time_series_mtex.LimeImageExplainer(
                feature_selection=cfg['xai']['lime']['feature_selection'],
                random_state=cfg['xai']['lime']['random_state']
            )

            explanation = explainer.explain_instance(
                image=x[0],
                classifier_fn=model.predict,
                class_to_explain=idx_to_explain,
                hide_color=cfg['xai']['lime']['hide_color'],
                top_labels=cfg['xai']['lime']['top_labels'],
                num_samples=cfg['xai']['lime']['num_samples'],
                model_regressor=None
            )

            temp, mask, activation = explanation.get_image_and_mask(
                positive_only=True,
                negative_only=False,
                num_features=cfg['xai']['lime']['num_features'],
                hide_rest=True
            )

            util_report.plot_heatmap_lime(outdir=reports_dir_lime, x=x, mask=mask, info='heatmap_lime')

        counter += 1

print("May be the force with you.")