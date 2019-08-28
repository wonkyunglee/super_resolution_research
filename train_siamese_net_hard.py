import os
import tqdm
import argparse
import pprint

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np
import skimage
import os
import glob
from skimage.io import imread
import skimage
import math
import time
#from utils.imresize import imresize
from datasets import get_train_dataloader, get_valid_dataloader, get_test_dataloader
from models import get_model
from losses import get_loss
from optimizers import get_optimizer
from schedulers import get_scheduler
from visualizers import get_visualizer
from tensorboardX import SummaryWriter

import utils.config
import utils.checkpoint
from utils.utils import quantize
from utils.metrics import get_psnr

device = None
model_type = None

def adjust_learning_rate(config, epoch):
    lr = config.optimizer.params.lr * (0.5 ** (epoch // config.scheduler.params.step_size))
    return lr

def train_single_epoch(config, model, dataloader, criterion,
                       optimizer, epoch, writer,
                       visualizer, postfix_dict):
    model.train()
    batch_size = config.train.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    log_dict = {}

    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    for i, (LR_patch, HR_patch, filepath) in tbar:
        shake= 2
        loss = 0
        diff = 0
        HR_patch_same = HR_patch.to(device)[:,:,:-shake, :-shake]
        LR_patch_same = LR_patch.to(device)[:,:,:-shake//2, :-shake//2]
        label = 1

        pred_dict_lr = model.forward(LR=LR_patch_same)
        pred_dict_hr = model.forward(HR=HR_patch_same)
        loss_dict = criterion['train'](pred_dict_hr=pred_dict_hr,
                                  pred_dict_lr=pred_dict_lr,
                                  label=label)
        loss += loss_dict['loss']
        diff += loss_dict['diff']


        HR_patch_diff1 = HR_patch.to(device)[:,:,shake:, shake:]
        LR_patch_diff1 = LR_patch.to(device)[:,:,:-shake//2, :-shake//2]
        label = 0
        pred_dict_lr = model.forward(LR=LR_patch_diff1)
        pred_dict_hr = model.forward(HR=HR_patch_diff1)
        loss_dict = criterion['train'](pred_dict_hr=pred_dict_hr,
                                  pred_dict_lr=pred_dict_lr,
                                  label=label)
        loss += loss_dict['loss']
        diff += loss_dict['diff']


        HR_patch_diff2 = HR_patch.to(device)[:,:,:-shake, :-shake]
        LR_patch_diff2 = LR_patch.to(device)[:,:,shake//2:, shake//2:]
        label = 0
        pred_dict_lr = model.forward(LR=LR_patch_diff2)
        pred_dict_hr = model.forward(HR=HR_patch_diff2)
        loss_dict = criterion['train'](pred_dict_hr=pred_dict_hr,
                                  pred_dict_lr=pred_dict_lr,
                                  label=label)
        loss += loss_dict['loss']
        diff += loss_dict['diff']

        HR_patch_diff3 = HR_patch.to(device)[:,:,:-shake, shake:]
        LR_patch_diff3 = LR_patch.to(device)[:,:,:-shake//2, :-shake//2]
        label = 0
        pred_dict_lr = model.forward(LR=LR_patch_diff3)
        pred_dict_hr = model.forward(HR=HR_patch_diff3)
        loss_dict = criterion['train'](pred_dict_hr=pred_dict_hr,
                                  pred_dict_lr=pred_dict_lr,
                                  label=label)
        loss += loss_dict['loss']
        diff += loss_dict['diff']


        HR_patch_diff4 = HR_patch.to(device)[:,:,:-shake, :-shake]
        LR_patch_diff4 = LR_patch.to(device)[:,:,:-shake//2, shake//2:]
        label = 0
        pred_dict_lr = model.forward(LR=LR_patch_diff4)
        pred_dict_hr = model.forward(HR=HR_patch_diff4)
        loss_dict = criterion['train'](pred_dict_hr=pred_dict_hr,
                                  pred_dict_lr=pred_dict_lr,
                                  label=label)
        loss += loss_dict['loss']
        diff += loss_dict['diff']

        HR_patch_diff5 = HR_patch.to(device)[:,:,shake:, :-shake]
        LR_patch_diff5 = LR_patch.to(device)[:,:,:-shake//2, :-shake//2]
        label = 0
        pred_dict_lr = model.forward(LR=LR_patch_diff5)
        pred_dict_hr = model.forward(HR=HR_patch_diff5)
        loss_dict = criterion['train'](pred_dict_hr=pred_dict_hr,
                                  pred_dict_lr=pred_dict_lr,
                                  label=label)
        loss += loss_dict['loss']
        diff += loss_dict['diff']


        HR_patch_diff6 = HR_patch.to(device)[:,:,:-shake, :-shake]
        LR_patch_diff6 = LR_patch.to(device)[:,:,shake//2:, :-shake//2]
        label = 0
        pred_dict_lr = model.forward(LR=LR_patch_diff6)
        pred_dict_hr = model.forward(HR=HR_patch_diff6)
        loss_dict = criterion['train'](pred_dict_hr=pred_dict_hr,
                                  pred_dict_lr=pred_dict_lr,
                                  label=label)
        loss += loss_dict['loss']
        diff += loss_dict['diff']


        loss.backward()
        optimizer.step()

        f_epoch = epoch + i / total_step
        log_dict['loss'] = loss.item()
        log_dict['diff'] = diff.item()

        log_dict['lr'] = optimizer.param_groups[0]['lr']
        for key, value in log_dict.items():
            postfix_dict['train/{}'.format(key)] = value

        desc = '{:5s}'.format('train')
        desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
        tbar.set_description(desc)
        tbar.set_postfix(**postfix_dict)

        if i % 100 == 0:
            log_step = int(f_epoch * 10000)
            if writer is not None:
                for key, value in log_dict.items():
                    writer.add_scalar('train/{}'.format(key), value, log_step)


def evaluate_single_epoch(config, model, dataloader, criterion,
                          epoch, writer, visualizer,
                          postfix_dict, eval_type):
    model.eval()
    with torch.no_grad():
        batch_size = config.eval.batch_size
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)

        total_psnr = 0
        total_loss = 0
        for i, (LR_img, HR_img, filepath) in tbar:
            loss = 0
            HR_img = HR_img[:,:1].to(device)
            LR_img = LR_img[:,:1].to(device)
            label = 1

            pred_dict_lr = model.forward(LR=LR_img)
            pred_dict_hr = model.forward(HR=HR_img)
            total_loss += criterion['val'](pred_dict_hr=pred_dict_hr,
                                     pred_dict_lr=pred_dict_lr,
                                     label=label)['loss']


            f_epoch = epoch + i / total_step
            desc = '{:5s}'.format(eval_type)
            desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
            tbar.set_description(desc)
            tbar.set_postfix(**postfix_dict)


        log_dict = {}
        avg_loss = total_loss.item() / (i+1)
        log_dict['loss'] = avg_loss

        for key, value in log_dict.items():
            if writer is not None:
                writer.add_scalar('{}/{}'.format(eval_type, key), value, epoch)
            postfix_dict['{}/{}'.format(eval_type, key)] = value

        return avg_loss


def train(config, model, dataloaders, criterion,
          optimizer, scheduler, writer, visualizer, start_epoch):
    num_epochs = config.train.num_epochs
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    postfix_dict = {'train/lr': 0.0,
                    'train/loss': 0.0,
                    'val/psnr': 0.0,
                    'val/loss': 0.0,
                    'test/psnr': 0.0,
                    'test/loss': 0.0}
    for epoch in range(start_epoch, num_epochs):

        # test phase
        evaluate_single_epoch(config, model, dataloaders['test'],
                              criterion, epoch, writer,
                              visualizer, postfix_dict,
                              eval_type='test')

        # val phase
        loss = evaluate_single_epoch(config, model, dataloaders['val'],
                                     criterion, epoch, writer,
                                     visualizer, postfix_dict,
                                     eval_type='val')
        if config.scheduler.name == 'reduce_lr_on_plateau':
            scheduler.step(loss)
        elif config.scheduler.name != 'reduce_lr_on_plateau':
            scheduler.step()

        utils.checkpoint.save_checkpoint(config, model, optimizer, epoch, 0,
                                         model_type=model_type)


        # train phase
        train_single_epoch(config, model, dataloaders['train'],
                           criterion, optimizer, epoch, writer,
                           visualizer, postfix_dict)




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(config):
    train_dir = config.train.dir
    model = get_model(config, model_type).to(device)
    print('The nubmer of parameters : %d'%count_parameters(model))
    criterion = get_loss(config)
    trainable_params = filter(lambda p: p.requires_grad,
                              model.parameters())
    optimizer = get_optimizer(config, trainable_params)

    checkpoint = utils.checkpoint.get_initial_checkpoint(config,
                                                         model_type=model_type)
    if checkpoint is not None:
        last_epoch, step = utils.checkpoint.load_checkpoint(model, optimizer,
                                    checkpoint, model_type=model_type)
    else:
        last_epoch, step = -1, -1

    print('from checkpoint: {} last epoch:{}'.format(checkpoint, last_epoch))
    scheduler = get_scheduler(config, optimizer, last_epoch)

    print(config.data)
    dataloaders = {'train':get_train_dataloader(config),
                   'val':get_valid_dataloader(config),
                   'test':get_test_dataloader(config)}

    writer = SummaryWriter(config.train[model_type + '_dir'])
    visualizer = get_visualizer(config)
    train(config, model, dataloaders, criterion, optimizer, scheduler,
          writer, visualizer, last_epoch+1)


def parse_args():
    parser = argparse.ArgumentParser(description='teacher network')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()


def main():
    global device
    import warnings
    warnings.filterwarnings("ignore")
    global model_type
    model_type = 'teacher'

    print('train %s network'%model_type)
    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')

    config = utils.config.load(args.config_file)

    os.environ["CUDA_VISIBLE_DEVICES"]= str(config.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pprint.PrettyPrinter(indent=2).pprint(config)
    utils.prepare_train_directories(config, model_type=model_type)
    run(config)

    print('success!')


if __name__ == '__main__':
    main()




