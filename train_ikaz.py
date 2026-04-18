import os
import time
import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
from pathlib import Path
import shutil

import torch
import torch.nn as nn
from torch import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from config import GlobalConfig
from model_ikaz import ai23
from dataloader import KarrDataset
import utility

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# BCE Dice Loss
def bce_dice_loss(Yp, Yt, smooth=1e-7):
    Yp = Yp.view(-1)
    Yt = Yt.view(-1)
    # BCE
    bce = F.binary_cross_entropy(Yp, Yt, reduction='mean')
    # dice loss
    intersection = (Yp * Yt).sum()
    dice_loss = 1 - ((2. * intersection + smooth) / (Yp.sum() + Yt.sum() + smooth))
    bce_dice_loss = bce + dice_loss
    return bce_dice_loss

# Renormalize loss weight based on Gradnorm
def renormalize_lw(current_lw, config):
    lw = np.array([tens.cpu().detach().numpy() for tens in current_lw])
    lws = np.array([lw[i][0] for i in range(len(lw))])

    # 1st algorithm in gradnorm's paper to renormalize
    coef = np.array(config.loss_weights).sum()/lws.sum()
    new_lws = [coef*lwx for lwx in lws]

    normalized_lws = [torch.cuda.FloatTensor([lw]).clone().detach().requires_grad_(True) for lw in new_lws]
    return normalized_lws

def train(model, dataloader, curr_epoch, optimizer, params_lw, optimizer_lw, device, writer, config):
    score = {
        'total_loss': AverageMeter(),
        'ss_loss': AverageMeter(),
        'wp_loss': AverageMeter(),
    }

    model.train()

    progress_bar = tqdm(total=len(dataloader))
    total_batch = len(dataloader)
    batch_idx = 0
    for data in dataloader:
        curr_step = curr_epoch * total_batch + batch_idx
        # Load then move to GPU
        rgbs = []
        segs = []
        pcd_xs = []
        pcd_zs = []
        for i in range(0, config.seq_len):
            rgbs.append(data['rgbs'][i].to(device, dtype=torch.float))
            segs.append(data['segs'][i].to(device, dtype=torch.float))
            pcd_xs.append(data['pcd_xs'][i].to(device, dtype=torch.float))
            pcd_zs.append(data['pcd_zs'][i].to(device, dtype=torch.float))
        rp1 = torch.stack(data['rp1'], dim=1).to(device, dtype=torch.float)
        rp2 = torch.stack(data['rp2'], dim=1).to(device, dtype=torch.float)
        gt_velocity = data['velocity'].to(device, dtype=torch.float)
        gt_wp = [torch.stack(data['waypoints'][j], dim=1).to(device, dtype=torch.float) for j in range(0, config.pred_len)]
        gt_wp = torch.stack(gt_wp, dim=1).to(device, dtype=torch.float)

        # forward pass
        pred_segs, pred_wp, _ = model(rgbs, pcd_xs, pcd_zs, rp1, rp2, gt_velocity)

        # compute loss
        seg_loss = 0
        for i in range(0, config.seq_len):
            seg_loss = seg_loss + bce_dice_loss(pred_segs[i], segs[i])
        seg_loss = seg_loss / config.seq_len
        wp_loss = F.l1_loss(pred_wp, gt_wp)
        total_loss = params_lw[0] * seg_loss + params_lw[1] * wp_loss

        # backpropagation, gradient calculation, and optimization
        optimizer.zero_grad()
        if batch_idx == 0:
            total_loss.backward()
            seg_loss_0 = torch.clone(seg_loss)
            wp_loss_0 = torch.clone(wp_loss)
        elif 0 < batch_idx < total_batch-1:
            total_loss.backward()
        elif batch_idx == total_batch - 1: # last batch 
            if config.MGN:
                optimizer_lw.zero_grad()
                total_loss.backward(retain_graph=True)
                # Takes gradient value from the 1st layer from each task-specified decoder and compute the gradient from the output layer till the bottleneck 
                params = list(filter(lambda p: p.requires_grad, model.parameters()))
                G0R = torch.autograd.grad(seg_loss, params[config.bottleneck[0]], retain_graph=True, create_graph=True)
                G0 = torch.norm(G0R[0], keepdim=True)
                G1R = torch.autograd.grad(wp_loss, params[config.bottleneck[1]], retain_graph=True, create_graph=True)
                G1 = torch.norm(G1R[0], keepdim=True)
                G_avg = (G0 + G1) / len(config.loss_weights)
                # calculate the relative loss
                seg_loss_hat = seg_loss / seg_loss_0
                wp_loss_hat = wp_loss / wp_loss_0
                loss_hat_avg = (seg_loss_hat + wp_loss_hat) / len(config.loss_weights)
                # calculate r_i_(t) relative inverse training rate for each task
                inv_rate_ss = seg_loss_hat / loss_hat_avg
                inv_rate_wp = wp_loss_hat / loss_hat_avg
                # calculate the constant of the target grad
                C0 = (G_avg * inv_rate_ss).detach()**config.lw_alpha
                C1 = (G_avg * inv_rate_wp).detach()**config.lw_alpha
                # calculate the total LGrad
                Lgrad = F.l1_loss(G0, C0) + F.l1_loss(G1, C1)
                # calculate the gradient loss based on the 2nd equation in Gradnorm's paper
                Lgrad.backward()
                # update loss weight
                optimizer_lw.step()
                lgrad = Lgrad.item()
                new_param_lw = optimizer_lw.param_groups[0]['params']
            else:
                total_loss.backward()
                lgrad = 0
                new_param_lw = 1
            
        optimizer.step()

        # calculate the avg loss and metrics for the processed batch
        score['total_loss'].update(total_loss.item())
        score['ss_loss'].update(seg_loss.item())
        score['wp_loss'].update(wp_loss.item())

        # summary writer
        writer.add_scalar('train/total_loss', total_loss.item(), curr_step)
        writer.add_scalar('train/ss_loss', seg_loss.item(), curr_step)
        writer.add_scalar('train/wp_loss', wp_loss.item(), curr_step)

        # progress bar
        postfix = OrderedDict([
            ('total_loss', score['total_loss'].avg),
            ('ss_loss', score['ss_loss'].avg),
            ('wp_loss', score['wp_loss'].avg)
        ])
        progress_bar.set_postfix(postfix)
        progress_bar.update(1)
        batch_idx += 1
    progress_bar.close()
    return postfix, new_param_lw, lgrad

def validate(model, dataloader, curr_epoch, device, writer, config):
    score = {
        'total_loss': AverageMeter(),
        'ss_loss': AverageMeter(),
        'wp_loss': AverageMeter(),
    }

    model.eval()
    
    with torch.no_grad():
        progress_bar = tqdm(total=len(dataloader))
        total_batch = len(dataloader)
        batch_idx = 0
        for data in dataloader:
            curr_step = curr_epoch * total_batch + batch_idx
            # Load then move to GPU
            rgbs = []
            segs = []
            pcd_xs = []
            pcd_zs = []
            for i in range(0, config.seq_len):
                rgbs.append(data['rgbs'][i].to(device, dtype=torch.float))
                segs.append(data['segs'][i].to(device, dtype=torch.float))
                pcd_xs.append(data['pcd_xs'][i].to(device, dtype=torch.float))
                pcd_zs.append(data['pcd_zs'][i].to(device, dtype=torch.float))
            rp1 = torch.stack(data['rp1'], dim=1).to(device, dtype=torch.float)
            rp2 = torch.stack(data['rp2'], dim=1).to(device, dtype=torch.float)
            gt_velocity = data['velocity'].to(device, dtype=torch.float)
            gt_wp = [torch.stack(data['waypoints'][j], dim=1).to(device, dtype=torch.float) for j in range(0, config.pred_len)]
            gt_wp = torch.stack(gt_wp, dim=1).to(device, dtype=torch.float)

            # forward pass
            pred_segs, pred_wp, _ = model(rgbs, pcd_xs, pcd_zs, rp1, rp2, gt_velocity)

            # compute loss
            seg_loss = 0
            for i in range(0, config.seq_len):
                seg_loss = seg_loss + bce_dice_loss(pred_segs[i], segs[i])
            seg_loss = seg_loss / config.seq_len
            wp_loss = F.l1_loss(pred_wp, gt_wp)
            total_loss = seg_loss + wp_loss

            # calculate the avg loss and metrics for the processed batch
            score['total_loss'].update(total_loss.item())
            score['ss_loss'].update(seg_loss.item())
            score['wp_loss'].update(wp_loss.item())

            # summary writer
            writer.add_scalar('val/total_loss', total_loss.item(), curr_step)
            writer.add_scalar('val/ss_loss', seg_loss.item(), curr_step)
            writer.add_scalar('val/wp_loss', wp_loss.item(), curr_step)

            # progress bar
            postfix = OrderedDict([
                ('total_loss', score['total_loss'].avg),
                ('ss_loss', score['ss_loss'].avg),
                ('wp_loss', score['wp_loss'].avg)
            ])
            progress_bar.set_postfix(postfix)
            progress_bar.update(1)
            batch_idx += 1
        progress_bar.close()
    return postfix

def main():
    config = GlobalConfig()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id

    print("Loading model architecture...")
    model = ai23(config, device).to(device, dtype=torch.float)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total trainable parameters: ', params)

    # optimizer configuration
    optima = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optima, mode='min', factor=0.5, patience=4, min_lr=1e-6)

    # load dataset
    print("Loading datasets...")
    train_set = KarrDataset(split='train')
    val_set = KarrDataset(split='val')
    print(f"  Train samples: {len(train_set)}")
    print(f"  Val samples:   {len(val_set)}")

    if len(train_set) % config.batch_size == 1:
        drop_last = True # so that not to mess up the MGN
    else:
        drop_last = False
    
    dataloader_train = DataLoader(
        train_set, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=drop_last
    )
    dataloader_val = DataLoader(
        val_set, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    if not os.path.exists(config.logdir+"/trainval_log.csv"):
        print("Begin Training...")
        os.makedirs(config.logdir, exist_ok=True)
        print('Created dir:', config.logdir)
        #optimizer lw
        params_lw = [torch.cuda.FloatTensor([config.loss_weights[i]]).clone().detach().requires_grad_(True) for i in range(len(config.loss_weights))]
        optima_lw = optim.SGD(params_lw, lr=config.lr)
        #set nilai awal
        curr_ep = 0
        lowest_score = float('inf')
        stop_count = config.init_stop_counter
    else:
        print('Continue training!!!!!!!!!!!!!!!!')
        print('Loading checkpoint from ' + config.logdir)
        #baca log history training sebelumnya
        log_trainval = pd.read_csv(config.logdir+"/trainval_log.csv")
        # replace variable2 ini
        # print(log_trainval['epoch'][-1:])
        curr_ep = int(log_trainval['epoch'][-1:]) + 1
        lowest_score = float(np.min(log_trainval['val_loss']))
        stop_count = int(log_trainval['stop_counter'][-1:])
        # Load checkpoint
        model.load_state_dict(torch.load(os.path.join(config.logdir, 'recent_model.pth')))
        optima.load_state_dict(torch.load(os.path.join(config.logdir, 'recent_optim.pth')))

        #set optima lw baru
        latest_lw = [float(log_trainval['lw_ss'][-1:]), float(log_trainval['lw_wp'][-1:]), float(log_trainval['lw_str'][-1:]), float(log_trainval['lw_thr'][-1:]), float(log_trainval['lw_brk'][-1:]), float(log_trainval['lw_redl'][-1:]), float(log_trainval['lw_stops'][-1:])]
        params_lw = [torch.cuda.FloatTensor([latest_lw[i]]).clone().detach().requires_grad_(True) for i in range(len(latest_lw))]
        optima_lw = optim.SGD(params_lw, lr=float(log_trainval['lrate'][-1:]))
        # optima_lw.param_groups[0]['lr'] = optima.param_groups[0]['lr'] # lr disamakan
        # optima_lw.load_state_dict(torch.load(os.path.join(config.logdir, 'recent_optim_lw.pth')))
        #update direktori dan buat tempat penyimpanan baru
        config.logdir += "/retrain"
        os.makedirs(config.logdir, exist_ok=True)
        print('Created new retrain dir:', config.logdir)
    
    shutil.copyfile('config.py', config.logdir+'/config.py')
    log = OrderedDict([
        ('epoch', []),
        ('best_model', []),
        ('val_loss', []),
        ('val_ss_loss', []),
        ('val_wp_loss', []),
        ('train_loss', []), 
        ('train_ss_loss', []),
        ('train_wp_loss', []),
        ('lrate', []),
        ('stop_counter', []), 
        ('lgrad_loss', []),
        ('lw_ss', []),
        ('lw_wp', []),
        ('elapsed_time', []),
    ])
    writer = SummaryWriter(log_dir=config.logdir)

    epoch = curr_ep
    while True:
        print("Epoch: {:05d}------------------------------------------------".format(epoch))
        #cetak lr dan lw
        if config.MGN:
            curr_lw = optima_lw.param_groups[0]['params']
            lw = np.array([tens.cpu().detach().numpy() for tens in curr_lw])
            lws = np.array([lw[i][0] for i in range(len(lw))])
            print("current loss weights: ", lws)    
        else:
            curr_lw = config.loss_weights
            lws = config.loss_weights
            print("current loss weights: ", config.loss_weights)
        print("current lr untuk training: ", optima.param_groups[0]['lr'])

        #training validation
        start_time = time.time() #waktu mulai
        train_log, new_params_lw, lgrad = train(model=model, dataloader=dataloader_train, curr_epoch=epoch, optimizer=optima, params_lw=curr_lw, optimizer_lw=optima_lw, device=device, writer=writer, config=config)
        val_log = validate(model=model, dataloader=dataloader_val, curr_epoch=epoch, device=device, writer=writer, config=config)
        writer.add_scalar('epoch/train/total_loss', train_log['total_loss'], epoch)
        writer.add_scalar('epoch/train/ss_loss', train_log['ss_loss'], epoch)
        writer.add_scalar('epoch/train/wp_loss', train_log['wp_loss'], epoch)
        writer.add_scalar('epoch/val/total_loss', val_log['total_loss'], epoch)
        writer.add_scalar('epoch/val/ss_loss', val_log['ss_loss'], epoch)
        writer.add_scalar('epoch/val/wp_loss', val_log['wp_loss'], epoch)
        if config.MGN:
            #update params lw yang sudah di renormalisasi ke optima_lw
            optima_lw.param_groups[0]['params'] = renormalize_lw(new_params_lw, config) #harus diclone supaya benar2 terpisah
            print("total loss gradient: "+str(lgrad))
        #update learning rate untuk training process
        scheduler.step(val_log['total_loss']) #parameter acuan reduce LR adalah val_total_metric
        optima_lw.param_groups[0]['lr'] = optima.param_groups[0]['lr'] #update lr disamakan
        elapsed_time = time.time() - start_time #hitung elapsedtime

        log['epoch'].append(epoch)
        log['lrate'].append(optima.param_groups[0]['lr'])
        log['train_loss'].append(train_log['total_loss'])
        log['val_loss'].append(val_log['total_loss'])
        log['train_ss_loss'].append(train_log['ss_loss'])
        log['val_ss_loss'].append(val_log['ss_loss'])
        log['train_wp_loss'].append(train_log['wp_loss'])
        log['val_wp_loss'].append(val_log['wp_loss'])
        log['lgrad_loss'].append(lgrad)
        log['lw_ss'].append(lws[0])
        log['lw_wp'].append(lws[1])
        log['elapsed_time'].append(elapsed_time)
        print('| t_total_l: %.4f | t_ss_l: %.4f | t_wp_l: %.4f |' % (train_log['total_loss'], train_log['ss_loss'], train_log['wp_loss']))
        print('| v_total_l: %.4f | v_ss_l: %.4f | v_wp_l: %.4f |' % (val_log['total_loss'], val_log['ss_loss'], val_log['wp_loss']))
        print('elapsed time: %.4f sec' % (elapsed_time))

        #save recent model dan optimizernya
        torch.save(model.state_dict(), os.path.join(config.logdir, 'recent_model.pth'))
        torch.save(optima.state_dict(), os.path.join(config.logdir, 'recent_optim.pth'))
        # torch.save(optima_lw.state_dict(), os.path.join(config.logdir, 'recent_optim_lw.pth'))

        #save model best only
        if val_log['total_loss'] < lowest_score:
            print("total_loss: %.4f < lowest sebelumnya: %.4f" % (val_log['total_loss'], lowest_score))
            print("model terbaik disave!")
            torch.save(model.state_dict(), os.path.join(config.logdir, 'best_model.pth'))
            torch.save(optima.state_dict(), os.path.join(config.logdir, 'best_optim.pth'))
            # torch.save(optima_lw.state_dict(), os.path.join(config.logdir, 'best_optim_lw.pth'))
            #v_total_l sekarang menjadi lowest_score
            lowest_score = val_log['total_loss']
            #reset stop counter
            stop_count = config.init_stop_counter
            print("stop counter direset ke: ", stop_count)
            #catat sebagai best model
            log['best_model'].append("BEST")
        else:
            print("v_total_l: %.4f >= lowest sebelumnya: %.4f" % (val_log['total_loss'], lowest_score))
            print("model tidak disave!")
            stop_count -= 1
            print("stop counter : ", stop_count)
            log['best_model'].append("")

        #update stop counter
        log['stop_counter'].append(stop_count)
        #paste ke csv file
        pd.DataFrame(log).to_csv(os.path.join(config.logdir, 'trainval_log.csv'), index=False)

        #kosongkan cuda chace
        torch.cuda.empty_cache()
        epoch += 1

        # early stopping jika stop counter sudah mencapai 0 dan early stop true
        if stop_count==0:
            print("TRAINING BERHENTI KARENA TIDAK ADA PENURUNAN TOTAL LOSS DALAM %d EPOCH TERAKHIR" % (config.init_stop_counter))
            break #loop

if __name__ == '__main__':
    main()


