import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./", "./src/xai"])
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import yaml
import datetime
import os
import torch
import numpy as np

from model.mtex import mtex_cnn

from src.utils import util_general
from src.utils import util_data
from src.utils import util_report


if __name__ == '__main__':

    print("Upload configuration file")
    with open('./configs/mtex_basicmotions.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Seed everything
    util_general.seed_all(cfg['seed'])

    # Parameters
    print("Parameters")
    worker = cfg['device']['worker']
    model_name = cfg['model']['model_name']
    classes = cfg['data']['classes']
    dataset_name = cfg['data']['dataset']

    # Device
    device = torch.device(worker if torch.cuda.is_available() else "cpu")
    num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
    print(f'device: {device}')

    # Files and Directories
    print("Files and directories")
    data_dir = os.path.join(cfg['data']['data_dir'], dataset_name)

    # Model dir
    model_dir = os.path.join(cfg['model']['model_dir'], dataset_name)
    util_general.create_dir(model_dir)

    # Report dir
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    reports_dir = os.path.join(cfg['reports']['reports_dir'], dataset_name, f"{cfg['exp_name']}_{time}")  # folder to save results
    util_general.create_dir(reports_dir)

    # Load dataset
    # Dataset listing -> https://www.timeseriesclassification.com/dataset.php
    print("Load dataset.")
    (X_train, y_train, X_test, y_test, y_train_nonencoded, y_test_nonencoded) = util_data.import_data(data_dir=data_dir, dataset_name=dataset_name, class_to_idx=classes)
    util_report.plot_mts(reports_dir, X_train[0], info='train_example')
    util_report.plot_mts(reports_dir, X_test[0], info='test_example')

    # Create the model
    model = mtex_cnn(input_shape=X_train.shape[1:], n_class=y_train.shape[1])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        X_train,
        y_train,
        epochs=cfg['trainer']["epochs"],
        batch_size=cfg['trainer']["batch_size"],
        verbose=1,
    )

    # Compute metrics
    acc_test = accuracy_score(y_test_nonencoded, np.argmax(model.predict(X_test), axis=1))
    f1_test = f1_score(y_test_nonencoded, np.argmax(model.predict(X_test), axis=1), average='macro')
    precision_test = precision_score(y_test_nonencoded, np.argmax(model.predict(X_test), axis=1), average='macro')
    recall_test = recall_score(y_test_nonencoded, np.argmax(model.predict(X_test), axis=1), average='macro')

    # Save metrics
    metrics = pd.DataFrame({'acc': acc_test, 'f1': f1_test, 'precision': precision_test, 'recall': recall_test}, index=[0])
    metrics.to_excel(os.path.join(reports_dir, 'metrics.xlsx'))

    # Save model
    model.save(os.path.join(model_dir, f"{model_name}_{dataset_name}.hdf5"))

print("May the force be with you!")