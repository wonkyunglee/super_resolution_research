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
1
from tensorboardX import SummaryWriter

import utils.config
import utils.checkpoint
from utils.utils import quantize
from utils.metrics import get_psnr

device = None

def adjust_learning_rate(config, epoch):
    lr = config.optimizer.params.lr * (0.5 ** (epoch // config.scheduler.params.step_size))
    return lr

def train_single_epoch(config, student_model, teacher_model, dataloader, criterion,
                       optimizer, epoch, writer, postfix_dict):
    student_model.eval() 
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
        
        atten, _,_,_ = teacher_model.forward(x=HR_patch)
        pred, _, _ = student_model.forward(lr=LR_patch, atten=atten)
        loss = criterion(pred, HR_patch)
        log_dict['loss'] = loss.item()

        loss.backward()
        if 'gradient_clip' in config.optimizer:
            lr = adjust_learning_rate(config, epoch)
            clip = config.optimizer.gradient_clip / lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        f_epoch = epoch + i / total_step

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


def evaluate_single_epoch(config, student_model, teacher_model, dataloader, 
                          criterion, epoch, writer, postfix_dict, eval_type):
    teacher_model.eval()
    student_model.eval()
    with torch.no_grad():
        batch_size = config.eval.batch_size
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)

        total_psnr = 0
        total_psnr_bic = 0
        total_loss = 0
        for i, (LR_img, HR_img, filepath) in tbar:
            HR_img = HR_img[:,:1].to(device)
            LR_img = LR_img[:,:1].to(device)

            atten, _,_,_ = teacher_model.forward(x=HR_img)
            pred, _, _ = student_model.forward(lr=LR_img, atten=atten)
            total_loss += criterion(pred, HR_img).item()

            pred = quantize(pred, config.data.rgb_range)
            total_psnr += get_psnr(pred, HR_img, config.data.scale,
                                  config.data.rgb_range, 
                                  benchmark=eval_type=='test')

            f_epoch = epoch + i / total_step
            desc = '{:5s}'.format(eval_type)
            desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
            tbar.set_description(desc)
            tbar.set_postfix(**postfix_dict)

        log_dict = {}
        avg_loss = total_loss / (i+1)
        avg_psnr = total_psnr / (i+1)
        log_dict['loss'] = avg_loss
        log_dict['psnr'] = avg_psnr

        for key, value in log_dict.items():
            if writer is not None:
                writer.add_scalar('{}/{}'.format(eval_type, key), value, epoch)
            postfix_dict['/{}'.format(eval_type, key)] = value

        return avg_psnr


def train(config, student_model, teacher_model, dataloaders, criterion,
          optimizer, scheduler, writer, start_epoch):
    num_epochs = config.train.num_epochs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.cuda()
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
        evaluate_single_epoch(config, student_model, teacher_model, 
                              dataloaders['test'],
                              criterion, epoch, writer, postfix_dict,
                              eval_type='test')
        
        # val phase
        psnr = evaluate_single_epoch(config, student_model, teacher_model,
                                     dataloaders['val'],
                                     criterion, epoch, writer, postfix_dict, 
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
        train_single_epoch(config, student_model, teacher_model, 
                           dataloaders['train'],
                           criterion, optimizer, epoch, writer, postfix_dict)


    return {'psnr': best_psnr, 'psnr_mavg': best_psnr_mavg}


def run(config):
    train_dir = config.train.dir

    hallucination_model = get_model(config, 'hallucination').cuda()
    student_model = get_model(config, 'student').cuda()
    criterion = get_loss(config)
    
    # for student
    optimizer = get_optimizer(config, student_model.parameters())
    checkpoint = utils.checkpoint.get_initial_checkpoint(config,
                                                         model_type='student')
    if checkpoint is not None:
        last_epoch, step = utils.checkpoint.load_checkpoint(student_model, 
                                 optimizer, checkpoint, model_type='student')
    else:
        last_epoch, step = -1, -1
    print('student model from checkpoint: {} last epoch:{}'.format(
        checkpoint, last_epoch))
    
    # for teacher
    optimizer = get_optimizer(config, teacher_model.parameters())
    checkpoint = utils.checkpoint.get_initial_checkpoint(config,
                                                         model_type='teacher')
    if checkpoint is not None:
        last_epoch, step = utils.checkpoint.load_checkpoint(teacher_model, 
                                 optimizer, checkpoint, model_type='teacher')
    else:
        last_epoch, step = -1, -1
    print('teacher model from checkpoint: {} last epoch:{}'.format(
        checkpoint, last_epoch))
    
    
    scheduler = get_scheduler(config, optimizer, last_epoch)

    print(config.data)
    dataloaders = {'train':get_train_dataloader(config),
                   'val':get_valid_dataloader(config),
                   'test':get_test_dataloader(config)}
    writer = SummaryWriter(config.train['teacher' + '_dir'])
    train(config, student_model, teacher_model, dataloaders,
          criterion, optimizer, scheduler, writer, last_epoch+1)


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

    print('train teacher network')
    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')

    config = utils.config.load(args.config_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # os.environ["CUDA_VISIBLE_DEVICES"]= str(config.gpu)

    pprint.PrettyPrinter(indent=2).pprint(config)
    utils.prepare_train_directories(config, model_type='student')
    run(config)

    print('success!')


if __name__ == '__main__':
    main()



