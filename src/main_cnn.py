import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./", "./src/xai"])

import os
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

from xai import gradcam_cnn, lime_time_series_cnn, ig_cnn, baseline_ig

if __name__ == '__main__':

    print("Upload configuration file")
    with open('./configs/xai_generali.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Seed everything
    util_general.seed_all(cfg['seed'])

    # Parameters
    print("Parameters")
    worker = cfg['device']['worker']
    model_name = cfg['model']['model_name']
    dataset_name = cfg['data']['dataset']
    modes = list(cfg['data']['modes'].keys())
    steps = ['train', 'valid', 'test']
    classes = cfg['data']['classes']

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
    reports_dir = os.path.join(cfg['reports']['reports_dir'], dataset_name, f"xai_{time}")  # folder to save results
    util_general.create_dir(reports_dir)

    # Upload pretrained models (tensorflow)
    print("Load pretrained model")
    model = util_model.build_model_cnn(
        model_dir=model_dir, model_name=model_name, samples_acc=cfg['data']['modes']['acc']['timestamp'], samples_pos=cfg['data']['modes']['pos']['timestamp']
    )

    # Data
    print("Create dataloader")
    fold_data = {step: pd.read_csv(os.path.join(fold_dir, '%s.txt' % step), delimiter=" ", index_col=0) for step in
                 steps}
    datasets_acc = {
        step: util_data.DatasetGenerali(data_dir=os.path.join(data_dir, modes[0], step), data=fold_data[step], mode=modes[0], class_to_idx=classes) for step  in steps
    }
    datasets_pos = {
        step: util_data.DatasetGenerali(data_dir=os.path.join(data_dir, modes[1], step), data=fold_data[step], mode= modes[1], class_to_idx=classes) for step  in steps
    }
    datasets = {
        step: util_data.MultimodalDataset(dataset_mode_1=datasets_acc[step], dataset_mode_2=datasets_pos[step]) for step in steps
    }

    # Data loaders
    data_loaders = {
        'train': torch.utils.data.DataLoader(
                datasets['train'], batch_size=1, shuffle=False, drop_last=True,  num_workers=num_workers, worker_init_fn=util_data.seed_worker
        ),
        'valid': torch.utils.data.DataLoader(
            datasets['valid'], batch_size=1, shuffle=False, num_workers=num_workers,   worker_init_fn=util_data.seed_worker
        ),
        'test': torch.utils.data.DataLoader(
            datasets['test'], batch_size=1, shuffle=False, num_workers=num_workers,   worker_init_fn=util_data.seed_worker
        )
    }
    # Define an iterator on dataloder
    data_iterator = iter(data_loaders[cfg['xai']['step_to_explain']])

    # Iterate over the dataset
    counter = 0
    while True:
        if counter > cfg['xai']['n_samples_to_explain']:
            break

        X_acc, X_pos, y, file_names = next(data_iterator)

        if cfg['xai']['list_samples_to_explain'] is not None:
            if file_names[0] not in cfg['xai']['list_samples_to_explain']:
                continue

        reports_dir_current = os.path.join(reports_dir, file_names[0])
        util_general.create_dir(reports_dir_current)

        if cfg['reports']['snap']:
            util_report.generali_plot_acc(outdir=reports_dir_current, acc=X_acc, info=f'mts')
            util_report.generali_plot_vel(outdir=reports_dir_current, vel=X_pos, info=f'vel')
        # To Tensoflow (the network was deployed from the company Tensorflow)
        X_acc = tf.convert_to_tensor(X_acc.numpy())
        X_pos = tf.convert_to_tensor(X_pos.numpy())

        # Define the class to explain.
        class_to_explain = util_xai.select_class_to_explain(cfg['xai']['class_to_explain'], X_acc, X_pos, y)


        # Gradcam
        if 'gradcam' in cfg['xai']['methods']:
            reports_dir_gradcam =  os.path.join(reports_dir_current, 'gradcam')
            os.makedirs(reports_dir_gradcam)

            gCAM = gradcam_cnn.GradCAM(model)
            gCAM.make_models()

            heatmap_original, crash_probability_original = gCAM.run_gradcam(
                acc=X_acc,
                vel=X_pos,
                repeats=True,
                relu=False,
                normalize=False,
                class_to_explain=class_to_explain
            )

            heatmap_perturbed_tilde = np.ones(X_acc[0].shape)
            gradcam_heatmap = np.ones(X_acc[0].shape)
            for ax, heatmap_name in zip(np.arange(3), ['x_axis', 'y_axis', 'z_axis']):
                # Perturb multivariate series.
                acc_gradcam = gCAM.perturb_time_series(X_acc, axis_to_explain=ax)

                # Run GradCAM on perturbed multivariate series.
                heatmap_perturbed_tilde[:, ax], crash_probability_perturbed = gCAM.run_gradcam(
                    acc=acc_gradcam,
                    vel=X_pos,
                    repeats=True,
                    relu=False,
                    normalize=True,
                    class_to_explain=class_to_explain
                )

                gradcam_heatmap[:, ax] = gCAM.ricombination_method(
                    heatmap_original,
                    crash_probability_original,
                    heatmap_perturbed_tilde[:, ax],
                    crash_probability_perturbed
                )

                if cfg['reports']['snap']:
                    util_report.generali_plot_acc(outdir=reports_dir_gradcam, acc=acc_gradcam, info=f'mts_{heatmap_name}')
                    util_report.generali_plot_acc_heatmap(outdir=reports_dir_gradcam, acc=acc_gradcam, heatmap=heatmap_perturbed_tilde[:, ax], info=f'mts_heatmap_{heatmap_name}')

            util_report.generali_plot_heatmap(outdir=reports_dir_gradcam, acc=X_acc, heatmap=gradcam_heatmap, info=f'gradcam')

        # Integrated gradients
        if 'ig' in cfg['xai']['methods']:
            reports_dir_ig = os.path.join(reports_dir_current, 'ig')
            os.makedirs(reports_dir_ig)

            ig_baseline = baseline_ig.BaselineIntegratedGradientsCNN(datasets['valid'], model, compute_mean=True)
            ig_method = ig_cnn.IntegratedGradients(model, m_steps=cfg['xai']['ig']['m_steps'])

            if class_to_explain == 'crash':
                baseline = ig_baseline.choose_baseline(cfg['xai']['ig']['baseline_crash'], input=X_acc, const_value=cfg['xai']['ig']['const_value_acc'])
                baseline_vel = ig_baseline.choose_baseline(cfg['xai']['ig']['baseline_crash'], input=X_pos, const_value=cfg['xai']['ig']['const_value_vel'])
            else:
                [baseline, baseline_vel] = ig_baseline.choose_baseline(cfg['xai']['ig']['baseline_non_crash'])

            if cfg['reports']['snap']:
                util_report.generali_plot_acc(outdir=reports_dir_ig, acc=baseline, info=f'baseline_acc')
                util_report.generali_plot_vel(outdir=reports_dir_ig, vel=baseline_vel, info=f'baseline_pos')

            ig_heatmap, delta, convergence = ig_method.integrated_gradients(X_acc, X_pos, baseline, baseline_vel, class_to_explain, normalize=True)

            util_report.generali_plot_heatmap(outdir=reports_dir_ig, acc=X_acc, heatmap=ig_heatmap,  info=f'ig')
            print('network activation using the signal and the baseline: ', delta)

        # LIME
        if 'lime' in cfg['xai']['methods']:
            reports_dir_lime = os.path.join(reports_dir_current, 'lime')
            os.makedirs(reports_dir_lime)
            explainer = lime_time_series_cnn.LimeImageExplainer(
                feature_selection=cfg['xai']['lime']['feature_selection'],
                random_state=cfg['xai']['lime']['random_state']
            )

            explanation = explainer.explain_instance(
                [X_acc[0], X_pos[0]],
                model.predict,
                class_to_explain,
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

            util_report.generali_plot_heatmap_lime(outdir=reports_dir_lime, acc=X_acc, mask=mask, info='heatmap_lime')
        counter += 1

print("May be the force with you.")