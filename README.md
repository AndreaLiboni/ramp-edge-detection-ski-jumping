# Ramp Edge Detector for Ski Jumping 

Fork of the code implementation of the paper "Deep Hough Transform for Semantic Line Detection" (ECCV 2020, PAMI 2021).
[arXiv2003.04676](https://arxiv.org/abs/2003.04676) | [GitHub page](https://github.com/Hanqer/deep-hough-transform)

## Getting Started
Create the virtual enviroment and source into it:
```bash
$ python -m venv venv
$ source venv/bin/activate
```
Install the required packages:
```bash
$ pip install -r requirements.txt
```
To install deep-hough package, run the following commands:
```bash
$ cd model/_cdht
$ python setup.py build 
$ python setup.py install --user
```

## Generate the dataset
This step require to already have an export in the CVAT format with the data.
Use the dataset_preparation.py file to prepare the dataset, and insert the correct paths in the lines 112, 113 and 116:
```bash
$ python dataset_preparation.py
```

## Training
Following the default config file 'config.yml', you can arbitrarily modify hyperparameters.
Then, run the following command.
```bash
$ python train.py
```
It is also possible to perform grid search, by running:
```bash
$ python grid_search.py
```
and also use the k-fold cross validation:
```bash
$ python train_kfold_cross_val.py
```

## Testing
Enable the 'TEST' flag in the 'config.yml' and use the command:
```bash
$ python train.py --model model.pth
```
