import argparse
import os
import random
import shutil
import time
from os.path import isfile, join, split

import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torch.optim
import tqdm
import yaml
from torch.optim import lr_scheduler
from logger import Logger

from dataloader import get_loader
from model.network import Net
from skimage.measure import label, regionprops
from tensorboardX import SummaryWriter
from utils import reverse_mapping, edge_align
from hungarian_matching import caculate_tp_fp_fn

import pandas as pd
import matplotlib.pyplot as plt

import cv2

# arguments from command line
parser = argparse.ArgumentParser(description='PyTorch Semantic-Line Training')
parser.add_argument('--config', default="./config.yml", help="path to config file")
parser.add_argument('--resume', default="", help='path to config file')
parser.add_argument('--tmp', default="", help='tmp folder to save results')
parser.add_argument('--model', default="", help='model checkpoint state file')
args = parser.parse_args()

# arguments from config file
CONFIGS = yaml.load(open(args.config), Loader=yaml.FullLoader)

# merge configs
if args.tmp != "" and args.tmp != CONFIGS["MISC"]["TMP"]:
    CONFIGS["MISC"]["TMP"] = args.tmp

CONFIGS["OPTIMIZER"]["WEIGHT_DECAY"] = float(CONFIGS["OPTIMIZER"]["WEIGHT_DECAY"])
CONFIGS["OPTIMIZER"]["LR_START"] = float(CONFIGS["OPTIMIZER"]["LR_START"])
CONFIGS["OPTIMIZER"]["LR_END"] = float(CONFIGS["OPTIMIZER"]["LR_END"])

os.makedirs(CONFIGS["MISC"]["TMP"], exist_ok=True)
logger = Logger(os.path.join(CONFIGS["MISC"]["TMP"], "log.txt"))

logger.info(CONFIGS)

def main():

    logger.info(args)
    assert os.path.isdir(CONFIGS["DATA"]["DIR"])

    if CONFIGS['TRAIN']['SEED'] is not None:
        random.seed(CONFIGS['TRAIN']['SEED'])
        torch.manual_seed(CONFIGS['TRAIN']['SEED'])
        cudnn.deterministic = True

    model = Net(
        dh_dimention=CONFIGS["MODEL"]["DH_DIMENTION"],
        backbone=CONFIGS["MODEL"]["BACKBONE"],
        num_conv_layer=CONFIGS["MODEL"]["NUM_CONV_LAYER"],
        num_pool_layer=CONFIGS["MODEL"]["NUM_POOL_LAYER"],
        num_fc_layer=CONFIGS["MODEL"]["NUM_FC_LAYER"],
    )
    
    if CONFIGS["TRAIN"]["DATA_PARALLEL"]:
        logger.info("Model Data Parallel")
        model = nn.DataParallel(model).cuda()

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIGS["OPTIMIZER"]["LR_START"],
        weight_decay=CONFIGS["OPTIMIZER"]["WEIGHT_DECAY"]
    )

    # learning rate scheduler
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=CONFIGS["OPTIMIZER"]["STEPS"], gamma=CONFIGS["OPTIMIZER"]["GAMMA"])
    scheduler = lr_scheduler.LinearLR(
        optimizer,
        start_factor=CONFIGS["OPTIMIZER"]["LR_START"],
        end_factor=CONFIGS["OPTIMIZER"]["LR_END"],
        total_iters=CONFIGS["TRAIN"]["EPOCHS"]
    )
    best_acc = 0
    start_epoch = 0
    train_epochs_loss = []
    train_epochs_acc = []
    test_epochs_loss = []
    test_epochs_acc = []
    if args.resume:
        if isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']

            train_epochs_loss = checkpoint['train_epochs_loss']
            train_epochs_acc = checkpoint['train_epochs_acc']
            test_epochs_loss = checkpoint['test_epochs_loss']
            test_epochs_acc = checkpoint['test_epochs_acc']

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
            return
    
    if args.model:
        if isfile(args.model):
            logger.info("=> loading model '{}'".format(args.model))
            try:
                model.load_state_dict(torch.load(args.model, weights_only=True))
            except:
                model.load_state_dict(torch.load(args.model)['state_dict'])
            logger.info("=> loaded model '{}'".format(args.model))
        else:
            logger.info("=> no model found at '{}'".format(args.model))

    # dataloader
    train_loader = get_loader(
        root_dir=CONFIGS["DATA"]["DIR"], 
        test=False,
        batch_size=CONFIGS["DATA"]["BATCH_SIZE"],
        shuffle=not CONFIGS["TRAIN"]["SHOW_DATASET"],
        num_workers=CONFIGS["DATA"]["WORKERS"]
    )
    test_loader = get_loader(
        root_dir=CONFIGS["DATA"]["DIR"], 
        test=True,
        batch_size=1,
        shuffle=False,
        num_workers=CONFIGS["DATA"]["WORKERS"]
    )

    logger.info("Data loading done.")

    # Tensorboard summary

    writer = SummaryWriter(log_dir=os.path.join(CONFIGS["MISC"]["TMP"]))

    is_best = False
    start_time = time.time()
    
    if CONFIGS["TRAIN"]["TEST"]:
        validate(test_loader, model, 0, writer, show_result=True)
        return
    
    if CONFIGS["TRAIN"]["SHOW_DATASET"]:
        visualize_save_path = os.path.join(CONFIGS["MISC"]["TMP"], 'visualize', 'dataset')
        os.makedirs(visualize_save_path, exist_ok=True)
        untransform = train_loader.dataset.untransform
        for data in train_loader:
            images, lines, names = data
            for model_input, line, name in zip(images, lines, names):
                img = untransform(model_input.cpu().detach()) * 255
                img = np.transpose(img.numpy(), (1, 2, 0))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                line_norm = line.cpu().detach().numpy() * 400
                img = cv2.line(img, (int(line_norm[0]), int(line_norm[1])), (int(line_norm[2]), int(line_norm[3])), (255, 0, 255), thickness=2)
                cv2.imwrite(os.path.join(visualize_save_path, name), img)
        return

    logger.info("Start training.")

    for epoch in range(start_epoch, CONFIGS["TRAIN"]["EPOCHS"]):
        
        train_loss, train_acc = train(train_loader, model, optimizer, epoch, writer)
        test_loss, test_acc = validate(test_loader, model, epoch, writer)

        train_epochs_loss.append(train_loss)
        train_epochs_acc.append(train_acc)
        test_epochs_loss.append(test_loss)
        test_epochs_acc.append(test_acc)
        #return

        scheduler.step()
        if best_acc < test_acc:
            is_best = True
            best_acc = test_acc
        else:
            is_best = False
        
        print('last_lr:' + str(scheduler.get_last_lr()))
        print('best_acc:' + str(best_acc))
        print('loss:' + str(train_loss))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            'train_epochs_loss': train_epochs_loss,
            'train_epochs_acc': train_epochs_acc,
            'test_epochs_loss': test_epochs_loss,
            'test_epochs_acc': test_epochs_acc
        }, is_best, path=CONFIGS["MISC"]["TMP"])
        
        # model.load_state_dict(torch.load(os.path.join(CONFIGS["MISC"]["TMP"], 'model_best.pth'), weights_only=True))

        t = time.time() - start_time       
        elapsed = DayHourMinute(t)
        t /= (epoch + 1) - start_epoch    # seconds per epoch
        t = (CONFIGS["TRAIN"]["EPOCHS"] - epoch - 1) * t
        remaining = DayHourMinute(t)
        
        logger.info("Epoch {0}/{1} finishied, auxiliaries saved to {2} .\t"
                    "Elapsed {elapsed.days:d} days {elapsed.hours:d} hours {elapsed.minutes:d} minutes.\t"
                    "Remaining {remaining.days:d} days {remaining.hours:d} hours {remaining.minutes:d} minutes.".format(
                    epoch, CONFIGS["TRAIN"]["EPOCHS"], CONFIGS["MISC"]["TMP"], elapsed=elapsed, remaining=remaining))

        df = pd.DataFrame([loss if loss < 0.002 else 0.002 for loss in train_epochs_loss]).plot()
        plt.xlabel('Epoch ' + str(epoch))
        plt.ylabel('Loss (max 0.001)')
        plt.savefig('model_output/loss.png')
        plt.close()
        
        df = pd.DataFrame(test_epochs_acc).plot()
        plt.xlabel('Epoch ' + str(epoch))
        plt.ylabel('Accuracy (%)')
        plt.savefig('model_output/accuracy.png')
        plt.close()

        if CONFIGS["TRAIN"]["COMPUTE_ACC"]:
            df = pd.DataFrame(train_epochs_acc).plot()
            plt.xlabel('Epoch ' + str(epoch))
            plt.ylabel('Accuracy (%)')
            plt.savefig('model_output/accuracy_train.png')
            plt.close()
        

    logger.info("Optimization done, ALL results saved to %s." % CONFIGS["MISC"]["TMP"])

def train(train_loader, model, optimizer, epoch, writer):
    # switch to train mode
    model.train()
    # torch.cuda.empty_cache()
    bar = tqdm.tqdm(train_loader)
    iter_num = len(train_loader.dataset) // CONFIGS["DATA"]["BATCH_SIZE"]

    total_loss = 0

    total_tp = np.zeros(1)
    total_fp = np.zeros(1)
    total_fn = np.zeros(1)

    total_tp_align = np.zeros(1)
    total_fp_align = np.zeros(1)
    total_fn_align = np.zeros(1)

    criterion = nn.MSELoss()
    untransform = train_loader.dataset.untransform
    for i, data in enumerate(bar):
        optimizer.zero_grad()
        images, lines, names = data

        if CONFIGS["TRAIN"]["DATA_PARALLEL"]:
            images = images.cuda()
            lines = lines.cuda()

        predicted_lines = model(images)

        loss = criterion(predicted_lines, lines)

        if not torch.isnan(loss):
            total_loss += loss.item()
        else:
            logger.info("Warnning: loss is Nan.")

        #record loss
        bar.set_description('Training Loss:{}'.format(loss.item()))

        # compute accuracy
        if CONFIGS["TRAIN"]["COMPUTE_ACC"]:
            b_points = [[point * 400 for point in predicted_line] for predicted_line in predicted_lines.detach().cpu()]
            gt_coords = [[point * 400 for point in line] for line in lines.detach().cpu()]

            for j in range(1, 1):
                tp, fp, fn = caculate_tp_fp_fn(b_points, gt_coords, thresh=j*0.01)
                total_tp[j-1] += tp
                total_fp[j-1] += fp
                total_fn[j-1] += fn
        
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        if i % CONFIGS["TRAIN"]["PRINT_FREQ"] == 0:
            visualize_save_path = os.path.join(CONFIGS["MISC"]["TMP"], 'visualize', 'train', str(epoch))
            os.makedirs(visualize_save_path, exist_ok=True)
            
            # Do visualization.
            for model_input, predicted_line, line, name in zip(images, predicted_lines, lines, names):
                img = untransform(model_input.cpu().detach()) * 255
                img = np.transpose(img.numpy(), (1, 2, 0))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                predicted_line_norm = predicted_line.cpu().detach().numpy() * 400
                line_norm = line.cpu().detach().numpy() * 400
                img = cv2.line(img, (int(predicted_line_norm[0]), int(predicted_line_norm[1])), (int(predicted_line_norm[2]), int(predicted_line_norm[3])), (255, 255, 0), thickness=2)
                img = cv2.line(img, (int(line_norm[0]), int(line_norm[1])), (int(line_norm[2]), int(line_norm[3])), (255, 0, 255), thickness=2)
                cv2.imwrite(os.path.join(visualize_save_path, name), img)
                break
    
    total_loss /= iter_num

    total_recall = total_tp / (total_tp + total_fn + 1e-8)
    total_precision = total_tp / (total_tp + total_fp + 1e-8)
    f = 2 * total_recall * total_precision / (total_recall + total_precision + 1e-8)
    acc = f.mean()
    
    logger.info('Train result: ==== Precision: %.5f, Recall: %.5f' % (total_precision.mean(), total_recall.mean()))
    logger.info('Train result: ==== F-measure: %.5f' % acc.mean())
    return total_loss, acc.mean()
 
    
def validate(test_loader, model, epoch, writer, show_result=False):
    # switch to evaluate mode
    model.eval()
    total_loss = 0

    total_tp = np.zeros(99)
    total_fp = np.zeros(99)
    total_fn = np.zeros(99)

    total_tp_align = np.zeros(99)
    total_fp_align = np.zeros(99)
    total_fn_align = np.zeros(99)

    untransform = test_loader.dataset.untransform
    with torch.no_grad():
        bar = tqdm.tqdm(test_loader)
        iter_num = len(test_loader.dataset) // 1
        for i, data in enumerate(bar):

            images, lines, names = data

            if CONFIGS["TRAIN"]["DATA_PARALLEL"]:
                images = images.cuda()
                lines = lines.cuda()
                
            predicted_lines = model(images)

            loss = torch.nn.functional.mse_loss(predicted_lines, lines)
            original_loss = loss
            loss_item = loss.item()
            loss = original_loss

            if not torch.isnan(loss):
                total_loss += loss_item
            else:
                logger.info("Warnning: val loss is Nan.")

            b_points = [[point * 400 for point in predicted_lines[0].cpu()]]
            gt_coords = [[point * 400 for point in lines[0].cpu()]]

            for j in range(1, 100):
                tp, fp, fn = caculate_tp_fp_fn(b_points, gt_coords, thresh=j*0.01)
                total_tp[j-1] += tp
                total_fp[j-1] += fp
                total_fn[j-1] += fn

            if i == 0 or show_result:
                img = untransform(images[0].cpu().detach()) * 255
                img = np.transpose(img.numpy(), (1, 2, 0))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                predicted_line_norm = predicted_lines[0].cpu().detach().numpy() * 400
                line_norm = lines[0].cpu().detach().numpy() * 400
                img = cv2.line(img, (int(predicted_line_norm[0]), int(predicted_line_norm[1])), (int(predicted_line_norm[2]), int(predicted_line_norm[3])), (255, 255, 0), thickness=2)
                img = cv2.line(img, (int(line_norm[0]), int(line_norm[1])), (int(line_norm[2]), int(line_norm[3])), (255, 0, 255), thickness=2)
                
                visualize_save_path = os.path.join(CONFIGS["MISC"]["TMP"], 'visualize', 'test')
                os.makedirs(visualize_save_path, exist_ok=True)
                if not show_result:
                    cv2.imwrite(os.path.join(visualize_save_path, str(epoch) + '.jpg'), img)
                else:
                    cv2.imwrite(os.path.join(visualize_save_path, names[0]), img)
            
        total_loss /= iter_num
        
        total_recall = total_tp / (total_tp + total_fn + 1e-8)
        total_precision = total_tp / (total_tp + total_fp + 1e-8)
        f = 2 * total_recall * total_precision / (total_recall + total_precision + 1e-8)
        
       
        logger.info('Validation result: ==== Precision: %.5f, Recall: %.5f' % (total_precision.mean(), total_recall.mean()))
        acc = f.mean()
        logger.info('Validation result: ==== F-measure: %.5f' % acc.mean())
        logger.info('Validation result: ==== F-measure@0.95: %.5f' % f[95 - 1])
        
    return total_loss, acc.mean()


def save_checkpoint(state, is_best, path, filename='checkpoint.pth'):
    torch.save(state, os.path.join(path, filename))
    if is_best:
        torch.save(state['state_dict'], os.path.join(path, 'model_best.pth'))
    torch.save(state['state_dict'], os.path.join(path, 'model_last.pth'))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



class DayHourMinute(object):
  
  def __init__(self, seconds):
      
      self.days = int(seconds // 86400)
      self.hours = int((seconds- (self.days * 86400)) // 3600)
      self.minutes = int((seconds - self.days * 86400 - self.hours * 3600) // 60)


if __name__ == '__main__':
    main()
