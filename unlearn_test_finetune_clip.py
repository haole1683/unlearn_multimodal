import argparse
import os
import numpy as np
import random
import time
import datetime
import json
import logging
from pathlib import Path

import clip

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import utils
from scheduler import create_scheduler
from optim import create_optimizer

from utils import distributed_utils 
from utils import ori_utils 
from utils.data_utils import (
    jsonPoisonDataset, jsonDataset, create_loader, create_sampler, ImageTextDatasetFromSupervisedDataset,
    ToTensorTrans,  To244TensorTrans, create_simple_loader
)
from utils.clip_util import (
    clip_normalize
)
from test_attack_classify import test_zero_shot

def evalutate(model):
    test_cifar_10_result = test_zero_shot(model)
    return test_cifar_10_result

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler):
    # train
    model.train()  
    
    loss_image = nn.CrossEntropyLoss()
    loss_text = nn.CrossEntropyLoss()

    metric_logger = ori_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', ori_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('total_loss', ori_utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 20
    step_size = 100
    warmup_iterations = warmup_steps*step_size 

    
    scaler = GradScaler()
    
    for i,(image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # batch_size = data_loader.batch_size
        batch_size = len(image)
        image = image.to(device,non_blocking=True)   
        # idx = idx.to(device,non_blocking=True)   
        text = tokenizer(text, truncate=True).to(device)

        optimizer.zero_grad()
        image = clip_normalize(image)

        with autocast():
            logits_per_image, logits_per_caption = model(image, text)
            ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)
            total_loss = (loss_image(logits_per_image, ground_truth) + loss_text(logits_per_caption, ground_truth)) / 2
        
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()    
        
        metric_logger.update(total_loss=total_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)  
               
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logging.info(f"Averaged stats: {metric_logger.global_avg()}")     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  



def main(args=None):
    
    if args.distributed:
        distributed_utils.init_distributed_mode(args)   

    if distributed_utils.is_main_process():
        log_level = logging.INFO
        distributed_utils.setup_logging(os.path.join(args.output_dir, "out.log"), log_level)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + distributed_utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    #### Model #### 
    logging.info("Creating model")
    model, _ = clip.load(args.clip_model, device, jit=False)
    model = model.float()
    
    tokenizer = clip.tokenize

    start_epoch = 0

    args.freeze_encoder = 'text'    # freeze the encoder
    if args.freeze_encoder == 'image':
        freeze_encoder = model.visual
        for param in freeze_encoder.parameters():
            param.requires_grad = False
    elif args.freeze_encoder == 'text':
        freeze_encoder = model.transformer
        for param in freeze_encoder.parameters():
            param.requires_grad = False

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   
    
    # optimizer_dict = {"opt": "adamW", "lr": 1e-5, "betas": (0.9, 0.98), "eps": 1.0e-6, "weight_decay": 0.2}
    # schedular_dict = {"sched": "cosine", "lr": 1e-5, "epochs": 15, "min_lr": 1e-6, "decay_rate": 1, "warmup_lr": 1e-5, "save_freq": 1, "warmup_epochs": 1, "cooldown_epochs": 0}
    
    # arg_opt = utils.AttrDict(optimizer_dict)
    # optimizer = create_optimizer(arg_opt, model)
    # arg_sche = utils.AttrDict(schedular_dict)
    # lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.98), eps=1.0e-6, weight_decay=0.2)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
    
    #### Dataset #### 
    poisoned = False
    # # json_path = "/remote-home/songtianwei/research/unlearn_multimodal/data/laion-cat-with-index.json"
    # json_path = "/remote-home/songtianwei/research/unlearn_multimodal/data/laion_cifar10.json"
    # noise_path = "/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn/cat_noise_ori_RN50.pt"
    # if not poisoned:
    #     train_dataset = jsonDataset(json_path, img_transform = To244TensorTrans, contain_index=False)
    # else:
    #     train_dataset = jsonPoisonDataset(json_path, noise_path, img_transform = To244TensorTrans, contain_index=False)

    # This is for cifar10 to imageTextDataset
    train_dataset = ImageTextDatasetFromSupervisedDataset("CIFAR10", 'train', transform=To244TensorTrans)
    print("You are loading the dataset of cifar10 image-text pair dataset")
    
    train_loader = create_simple_loader(train_dataset)
    
    max_epoch = 40
    warmup_steps = 10

    logging.info("Start training")
    start_time = time.time()    
    for epoch in range(start_epoch, max_epoch):
        # if args.distributed:
        #     train_loader.sampler.set_epoch(epoch)
        result = evalutate(model)
        print(result)
        train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler)  
            
        
        if distributed_utils.is_main_process():  
            # save the model to local
            tgt_path = "/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_finetune_clip"
            clip_version = args.clip_model.replace('/','_')
            if poisoned:
                torch.save(model_without_ddp.state_dict(), os.path.join(tgt_path, f"model_{clip_version}_poison_epoch{epoch}.pth"))
            else:
                torch.save(model_without_ddp.state_dict(), os.path.join(tgt_path, f"model_{clip_version}_epoch{epoch}.pth"))        
           
        lr_scheduler.step(epoch+warmup_steps+1)  
        if args.distributed:
            dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f'Training time {total_time_str}') 

    if distributed_utils.is_main_process():   
        pass           

def run_default_experiment():
    parser = argparse.ArgumentParser()     
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action="store_true")

    # poisoning
    parser.add_argument('--clip_model', default='RN50', help="image encoder type of clip")
    parser.add_argument('--freeze_encoder', default='', help="image or text or none") # fi/ft = freeze image/text

    # config overload
    parser.add_argument('--output_dir', default='output/clip_poison_pascal_sheep2aeroplane_1.00/')
    args = parser.parse_args()

    args.output_dir = "/remote-home/songtianwei/research/unlearn_multimodal/output/finetune_clip"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()     

    parser.add_argument('--dataset', default='pascal')
    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--debug',  action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action="store_true")

    # poisoning
    parser.add_argument('--clip_model', default='RN50', help="image encoder type of clip", choices=['RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'ViT-B/16'])
    parser.add_argument('--freeze_encoder', default='', help="image or text or none") # fi/ft = freeze image/text

    # config overload
    parser.add_argument('--overload_config', action='store_true')
    parser.add_argument('--output_dir', default='output/clip_poison_pascal_sheep2aeroplane_1.00/')
    args = parser.parse_args()

    args.output_dir = "/remote-home/songtianwei/research/unlearn_multimodal/output/finetune_clip_RN50"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
