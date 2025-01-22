import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from skorch import NeuralNet
from torch import nn
from torch import optim
import torch
from yaml import safe_load

from hungarian_matching import caculate_tp_fp_fn
from model.network import Net
from dataloader import SkiTBDataset, transform, untransform

CONFIGS = safe_load(open('config.yml'))

def accuracy_score(y_true, y_pred):
    print("start accuracy_score")
    total_tp = np.zeros(99)
    total_fp = np.zeros(99)
    total_fn = np.zeros(99)

    total_tp_align = np.zeros(99)
    total_fp_align = np.zeros(99)
    total_fn_align = np.zeros(99)

    gt_points = [[point * 400 for point in line] for line in y_true]
    predicted_points = [[point * 400 for point in line] for line in y_pred]

    for j in range(1, 100):
        tp, fp, fn = caculate_tp_fp_fn(predicted_points, gt_points, thresh=j*0.01)
        total_tp[j-1] += tp
        total_fp[j-1] += fp
        total_fn[j-1] += fn
    
    total_recall = total_tp / (total_tp + total_fn + 1e-8)
    total_precision = total_tp / (total_tp + total_fp + 1e-8)
    f = 2 * total_recall * total_precision / (total_recall + total_precision + 1e-8)
    acc = f.mean()
    print("end accuracy_score")
    return acc

estimator = NeuralNet(
    module=Net,
    device='cuda',

    module__backbone='resnet50',

    optimizer=optim.AdamW,
    criterion=nn.MSELoss,
    max_epochs=5,
    batch_size=8,
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
    # if i > 100:
    #     break

x_train = np.array(x_train)
y_train = np.array(y_train)
print("data loaded!")

# grid search
param_grid = {
    'module__int_dim': [50, 100, 200],
}

grid = GridSearchCV(
    estimator=estimator,
    param_grid=param_grid,
    n_jobs=1,
    verbose=3,
    scoring=make_scorer(accuracy_score),
    error_score='raise'
)
grid_result = grid.fit(x_train, y_train)

print(grid_result.best_score_)
print(grid_result.best_params_)
print(grid_result)
