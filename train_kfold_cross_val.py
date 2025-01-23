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

from model.network import Net
from dataloader import SkiTBDataset, transform, untransform
from grid_search import accuracy_score

def main():
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
        max_epochs=CONFIGS["TRAIN"]["EPOCHS"],
        batch_size=CONFIGS["DATA"]["BATCH_SIZE"],
        lr=CONFIGS["OPTIMIZER"]["LR_START"],

    )

    # data
    train_dataset = SkiTBDataset(
        root_dir=CONFIGS["DATA"]["DIR"],
        test=False,
        transform=transform,
        untransform=untransform,
        use_augmentation=True,
    )

    print("loading data...")
    x_train = []
    y_train = []
    for i, (image, target, _) in enumerate(train_dataset):
        x_train.append(image.numpy())
        y_train.append(target.numpy())

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print("data loaded!")

    # training
    kfold = KFold(n_splits=5, shuffle=True)
    results = cross_val_score(
        estimator=model,
        X=x_train,
        y=y_train,
        cv=kfold,
        scoring=make_scorer(accuracy_score),
    )
    print("mean = %.3f; std = %.3f" % (results.mean(), results.std()))


    # saving
    joblib.dump(clf, join(CONFIGS["MISC"]["TMP"], "model.pkl"))

if __name__ == "__main__":
    main()
