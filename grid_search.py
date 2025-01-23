from comet_ml import start, login
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from skorch import NeuralNet
from torch import nn
from torch import optim
import torch
from yaml import safe_load

from hungarian_matching import caculate_tp_fp_fn
from model.network import Net
from dataloader import SkiTBDataset, transform, untransform
from utils import accuracy_score

def main():
    login()

    CONFIGS = safe_load(open('config.yml'))

    estimator = NeuralNet(
        module=Net,
        device='cuda',

        module__backbone='resnet50',
        module__dh_dimention=(100, 100),

        optimizer=optim.AdamW,
        criterion=nn.MSELoss,
        max_epochs=3,
        batch_size=8,
        lr=5e-3,
    )

    # data
    train_dataset = SkiTBDataset(
        root_dir=CONFIGS["DATA"]["DIR"],
        test=False,
        transform=transform,
        untransform=untransform,
        use_augmentation=False,
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

    # grid search
    param_grid = {
        # 'module__dh_dimention': [(50,50), (100,100), (200,200)],
        'module__num_conv_layer': [1, 2, 6],
        'module__num_pool_layer': [1, 2, 4],
        'module__num_fc_layer': [2, 4, 6],
        # 'lr': [5e-3, 5e-5, 5e-7],
    }

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        n_jobs=1,
        verbose=3,
        cv=3,
        scoring=make_scorer(accuracy_score),
        error_score='raise'
    )
    grid_result = grid.fit(x_train, y_train)

    # log experiment on comet
    for i in range(len(grid_result.cv_results_['params'])):
        exp = start(project_name='ramp-edge-detection-ski-jumping')

        for k,v in grid_result.cv_results_.items():
            if k == "params":
                exp.log_parameters(v[i])
            else:
                exp.log_metric(k,v[i])


if __name__ == "__main__":
    main()
