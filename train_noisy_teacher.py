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
from utils.metrics import get_psnr
from utils.utils import quantize

device = None
model_tyep = None

def adjust_learning_rate(config, epoch):
    lr = config.optimizer.params.lr * (0.5 ** (epoch // config.scheduler.params.step_size))
    return lr

def train_single_epoch(config, teacher_model, dataloader, criterion,
                       optimizer, epoch, writer, visualizer, postfix_dict):
    teacher_model.train()
    batch_size = config.train.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    log_dict = {}

    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    for i, (LR_patch, HR_patch, filepath) in tbar:
        HR_patch = HR_patch.to(device)
        LR_patch = LR_patch.to(device)

        optimizer.zero_grad()

        teacher_pred_dict = teacher_model.forward(LR=LR_patch, HR=HR_patch)
        loss = criterion['train'](teacher_pred_dict, HR_patch)

        for k, v in loss.items():
            log_dict[k] = v.item()
        loss['loss'].backward()
        optimizer.step()

        f_epoch = epoch + i / total_step

        log_dict['lr'] = optimizer.param_groups[0]['lr']
        for key, value in log_dict.items():
            if 'train/{}'.format(key) in postfix_dict:
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


def evaluate_single_epoch(config, teacher_model, dataloader,
                          criterion, epoch, writer,
                          visualizer, postfix_dict, eval_type):
    teacher_model.eval()
    with torch.no_grad():
        batch_size = config.eval.batch_size
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)

        total_psnr = 0
        total_loss = 0
        for i, (LR_img, HR_img, filepath) in tbar:
            HR_img = HR_img[:,:1].to(device)
            LR_img = LR_img[:,:1].to(device)

            teacher_pred_dict = teacher_model.forward(LR=LR_img, HR=HR_img)
            pred_hr = teacher_pred_dict['hr']
            total_loss += criterion['val'](pred_hr, HR_img).item()

            pred_hr = quantize(pred_hr, config.data.rgb_range)
            total_psnr += get_psnr(pred_hr, HR_img, config.data.scale,
                                  config.data.rgb_range,
                                  benchmark=eval_type=='test')

            f_epoch = epoch + i / total_step
            desc = '{:5s}'.format(eval_type)
            desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
            tbar.set_description(desc)
            tbar.set_postfix(**postfix_dict)


            if writer is not None and eval_type == 'test':
                fig = visualizer(LR_img, HR_img,
                                 teacher_pred_dict, teacher_pred_dict)
                writer.add_figure('{}/{:04d}'.format(eval_type, i), fig,
                                 global_step=epoch)

#         print(total_pseudo_psnr / (i+1))
        log_dict = {}
        avg_loss = total_loss / (i+1)
        avg_psnr = total_psnr / (i+1)
        log_dict['loss'] = avg_loss
        log_dict['psnr'] = avg_psnr

        for key, value in log_dict.items():
            if writer is not None:
                writer.add_scalar('{}/{}'.format(eval_type, key), value, epoch)
            postfix_dict['{}/{}'.format(eval_type, key)] = value

        return avg_psnr


def train(config, teacher_model, dataloaders, criterion,
          optimizer, scheduler, writer, visualizer, start_epoch):
    num_epochs = config.train.num_epochs
    if torch.cuda.device_count() > 1:
        teacher_model = torch.nn.DataParallel(teacher_model)

    postfix_dict = {'train/lr': 0.0,
                    'train/loss': 0.0,
                    'val/psnr': 0.0,
                    'val/loss': 0.0,
                    'test/psnr': 0.0,
                    'test/loss': 0.0}
    psnr_list = []
    best_psnr = 0.0
    best_psnr_mavg = 0.0
    for epoch in range(start_epoch, num_epochs):

        # test phase
        evaluate_single_epoch(config, teacher_model,
                              dataloaders['test'],
                              criterion, epoch, writer,
                              visualizer, postfix_dict,
                              eval_type='test')

        # val phase
        psnr = evaluate_single_epoch(config, teacher_model,
                                     dataloaders['val'],
                                     criterion, epoch, writer,
                                     visualizer, postfix_dict,
                                     eval_type='val')
        if config.scheduler.name == 'reduce_lr_on_plateau':
            scheduler.step(psnr)
        elif config.scheduler.name != 'reduce_lr_on_plateau':
            scheduler.step()

        utils.checkpoint.save_checkpoint(config, teacher_model, optimizer,
                                         epoch, 0,
                                         model_type='teacher')
        psnr_list.append(psnr)
        psnr_list = psnr_list[-10:]
        psnr_mavg = sum(psnr_list) / len(psnr_list)

        if psnr > best_psnr:
            best_psnr = psnr
        if psnr_mavg > best_psnr_mavg:
            best_psnr_mavg = psnr_mavg

        # train phase
        train_single_epoch(config, teacher_model,
                           dataloaders['train'],
                           criterion, optimizer, epoch, writer,
                           visualizer, postfix_dict)


    return {'psnr': best_psnr, 'psnr_mavg': best_psnr_mavg}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(config):
    teacher_model = get_model(config, 'teacher').to(device)
    criterion = get_loss(config)

    # for teacher
    trainable_params = filter(lambda p: p.requires_grad,
                              teacher_model.parameters())
    optimizer_t = get_optimizer(config, teacher_model.parameters())
    checkpoint_t = utils.checkpoint.get_initial_checkpoint(config,
                                                         model_type='teacher')
    if checkpoint_t is not None:
        last_epoch_t, step_t = utils.checkpoint.load_checkpoint(teacher_model,
                                 optimizer_t, checkpoint_t, model_type='teacher')
    else:
        last_epoch_t, step_t = -1, -1
    print('teacher model from checkpoint: {} last epoch:{}'.format(
        checkpoint_t, last_epoch_t))


    scheduler_t = get_scheduler(config, optimizer_t, last_epoch_t)

    print(config.data)
    dataloaders = {'train':get_train_dataloader(config),
                   'val':get_valid_dataloader(config),
                   'test':get_test_dataloader(config)}
    writer = SummaryWriter(config.train['teacher' + '_dir'])
    visualizer = get_visualizer(config)
    train(config, teacher_model, dataloaders,
          criterion, optimizer_t, scheduler_t, writer,
          visualizer, last_epoch_t+1)


def parse_args():
    parser = argparse.ArgumentParser(description='noisy teacher network')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()


def main():
    global device
    import warnings
    global model_type
    model_type = 'teacher'

    warnings.filterwarnings("ignore")

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



