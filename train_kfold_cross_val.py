from comet_ml import start, login
from comet_ml.integration.sklearn import log_model
import numpy as np
from skorch import NeuralNet
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer
from torch import nn
from torch import optim
import torch
from yaml import safe_load
import joblib
from os.path import join
import sys

from model.network import Net
from dataloader import SkiTBDataset, transform, untransform
from utils import accuracy_score, get_memory, memory_limit

def evaluate(y_test, y_pred):
    return {
        'accuracy': accuracy_score(y_test, y_pred),
    }

def main():
    login()
    CONFIGS = safe_load(open('config.yml'))
    CONFIGS["OPTIMIZER"]["LR_START"] = float(CONFIGS["OPTIMIZER"]["LR_START"])

    model = NeuralNet(
        module=Net,
        device='cuda' if CONFIGS["TRAIN"]["DATA_PARALLEL"] else 'cpu',

        module__backbone=CONFIGS["MODEL"]["BACKBONE"],
        module__dh_dimention=CONFIGS["MODEL"]["DH_DIMENTION"],
        module__num_conv_layer=CONFIGS["MODEL"]["NUM_CONV_LAYER"],
        module__num_pool_layer=CONFIGS["MODEL"]["NUM_POOL_LAYER"],
        module__num_fc_layer=CONFIGS["MODEL"]["NUM_FC_LAYER"],

        optimizer=optim.AdamW,
        criterion=nn.MSELoss,
        max_epochs=4,
        batch_size=CONFIGS["DATA"]["BATCH_SIZE"],
        lr=CONFIGS["OPTIMIZER"]["LR_START"],

    )

    # data
    train_dataset = SkiTBDataset(
        root_dir=CONFIGS["DATA"]["DIR"],
        test=False,
        transform=transform,
        untransform=untransform,
        set_augmentation=3,
    )
    test_dataset = SkiTBDataset(
        root_dir=CONFIGS["DATA"]["DIR"],
        test=True,
        transform=transform,
        untransform=untransform,
    )

    print("loading data...")
    print("train")
    x_train = []
    y_train = []
    for i, (image, target, _) in enumerate(train_dataset):
        x_train.append(image.numpy())
        y_train.append(target.numpy())

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    print("test")
    x_test = []
    y_test = []
    for i, (image, target, _) in enumerate(test_dataset):
        x_test.append(image.numpy())
        y_test.append(target.numpy())

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    print("data loaded!")

    exp = start(project_name='ramp-edge-detection-ski-jumping')

    # training
    kfold = KFold(n_splits=5, shuffle=True)
    results = cross_val_score(
        estimator=model,
        X=x_train,
        y=y_train,
        cv=kfold,
        scoring=make_scorer(accuracy_score),
        n_jobs=1
    )
    print("mean = %.3f; std = %.3f" % (results.mean(), results.std()))

    # saving
    joblib.dump(model, join(CONFIGS["MISC"]["TMP"], "model.pkl"))

    # logging to comet
    model = joblib.load(join(CONFIGS["MISC"]["TMP"], "model.pkl"))
    model.fit(x_train, y_train)

    # train
    y_train_pred = model.predict(x_train)
    with exp.train():
        metrics = evaluate(y_train, y_train_pred)
        exp.log_metrics(metrics)
    
    # test
    y_test_pred = model.predict(x_test)
    with exp.test():
        metrics = evaluate(y_test, y_test_pred)
        exp.log_metrics(metrics)

    # save model on comet
    log_model(
        exp,
        "my-model",
        model,
        persistence_module=joblib,
    )


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)
