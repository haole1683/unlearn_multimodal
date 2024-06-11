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
    jsonPoisonDataset, jsonDataset, create_loader, create_sampler, 
    ImageTextDatasetFromSupervisedDataset, ImageTextDatasetFromSupervisedDatasetPoison,
    ToTensorTrans,  To244TensorTrans, To288TensorTrans,
    create_simple_loader
)
from utils.clip_util import (
    clip_normalize
)
from utils.clip_util import (
    CustomCLIP
)
from utils.record_utils import (
    jsonRecord
)
from test_attack_classify import test_zero_shot_and_linear, evaluate_zero_shot_and_linear


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
    
    # print('Before training1')
    # evalutate(model, 'RN101')
    # model.eval()
    
    for i,(image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # batch_size = data_loader.batch_size
        batch_size = len(image)
        image = image.to(device,non_blocking=True)   
        # idx = idx.to(device,non_blocking=True)   
        text = clip.tokenize(text, truncate=True).to(device)
        # text = tokenizer(text, truncate=True).to(device)

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
    
    # print('After training1')
    # evalutate(model, 'RN101')
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logging.info(f"Averaged stats: {metric_logger.global_avg()}")     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  



def main(args=None):
    
    if args.distributed:
        distributed_utils.init_distributed_mode(args)   

    if distributed_utils.is_main_process():
        log_level = logging.INFO
        log_name = f"finetune_clip_dataset-{args.finetune_dataset}_{'poison' if args.poisoned else 'natural'}.log"
        distributed_utils.setup_logging(os.path.join(args.output_dir, log_name), log_level)

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
    
    if args.from_scratch:
        #### random init the model parameter
        for param in model.parameters():
            param.data = torch.randn_like(param.data) * 0.01
    model = model.float()
    
    tokenizer = clip.tokenize

    start_epoch = 0

    model = model.to(device)   
    
    # custom_model = CustomCLIP(model)
    
    name_to_update = "text_projection"
    for name, param in model.named_parameters():
        if name_to_update in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    # Double check
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")
    
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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1.0e-6, weight_decay=0.2)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
    
    finetune_dataset = args.finetune_dataset
    
    record_path = os.path.join(args.output_dir, f"record_{finetune_dataset}_{'poison' if args.poisoned else 'natural'}.json")
    myJsonRecord = jsonRecord(record_path)
    myJsonRecord.save_args(args)
    #### Dataset #### 
    # experiment 1
    if finetune_dataset == "myLaion":
        json_path = "./data/laion-truck.json"
        json_all_path = "./data/laion-all-with-index.json"
        
        noise_path = args.noise_path
        
        if args.clip_model == "RN50x4":
            the_transform = To288TensorTrans
        else:
            the_transform = To244TensorTrans
        
        if not args.poisoned:
            train_dataset = jsonDataset(json_all_path, img_transform = the_transform, contain_index=False)
        else:
            train_dataset = jsonPoisonDataset(json_all_path, noise_path, img_transform = the_transform, contain_index=False)


    # experiment 2
    # This is for cifar10 to imageTextDataset
    elif finetune_dataset == "cifar10":
        noise_path = "./output/train_g_unlearn/truck_noise_RN50.pt"
        if not args.poisoned:
            train_dataset = ImageTextDatasetFromSupervisedDataset("CIFAR10", 'train', transform=To244TensorTrans)
        else:
            train_dataset = ImageTextDatasetFromSupervisedDatasetPoison("CIFAR10", 'train', transform=To244TensorTrans, noise_path=noise_path)
        print("You are loading the dataset of cifar10 image-text pair dataset")
    
    train_loader = create_simple_loader(train_dataset, args)
    
    max_epoch = 40
    warmup_steps = 10

    logging.info("Start training")
    start_time = time.time()    
    for epoch in range(start_epoch, max_epoch):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        result = evaluate_zero_shot_and_linear(model, args.clip_model)
        result['epoch'] = epoch
        print(result)
        logging.info(f"Epoch {epoch}, result: {result}")
        myJsonRecord.save_exp_res(result)
        
        train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler)  
        
        if distributed_utils.is_main_process():  
            # save the model to local
            tgt_path = "./output/unlearn_finetune_clip"
            clip_version = args.clip_model.replace('/','_')
            if args.poisoned:
                torch.save(model_without_ddp.state_dict(), os.path.join(tgt_path, f"model_{clip_version}_poison_epoch_{epoch}.pth"))
            else:
                torch.save(model_without_ddp.state_dict(), os.path.join(tgt_path, f"model_{clip_version}_epoch_{epoch}.pth"))        
           
        lr_scheduler.step(epoch+warmup_steps+1)  
        if args.distributed:
            dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f'Training time {total_time_str}') 

    if distributed_utils.is_main_process():   
        pass           

            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()     

    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action="store_true")
    parser.add_argument('--poisoned', action="store_true")
    parser.add_argument('--finetune_dataset', default='myLaion')
    
    # training
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)

    # poisoning
    parser.add_argument('--clip_model', default='RN50', help="image encoder type of clip", choices=['RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'ViT-B/16'])
    parser.add_argument('--freeze_encoder', default='text', help="image or text or none", choices=['both','image','text','none']) # fi/ft = freeze image/text
    parser.add_argument('--from_scratch', action='store_true', help="train from scratch")
    
    # config overload
    parser.add_argument('--overload_config', action='store_true')
    parser.add_argument('--output_dir', default="./output/unlearn_stage3_test_clip_finetune_one_layer/")
    
    # noise
    parser.add_argument('--noise_path', default="./output/unlearn_stage2_generate_noise/RN101/noise_gen2_46221-224-224_all_RN101.pt")
    parser.add_argument('--test_train_type', default='finetune_clip')
    args = parser.parse_args()
    
    if args.from_scratch:
        args.output_dir = os.path.join(args.output_dir, "from_scratch")
    else:
        args.output_dir = os.path.join(args.output_dir, "from_pretrain")
    
    if args.poisoned:
        args.output_dir = os.path.join(args.output_dir, "poisoned")
        noise_path = args.noise_path
        noise_clip_version = noise_path.split('/')[-2]
        args.output_dir = os.path.join(args.output_dir, f"noise_of_{args.finetune_dataset}_{noise_clip_version}")
    else:
        args.output_dir = os.path.join(args.output_dir, "natural")
    clip_model_str = args.clip_model.replace('/','_')
    args.output_dir = os.path.join(args.output_dir, f"{args.finetune_dataset}_{clip_model_str}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
