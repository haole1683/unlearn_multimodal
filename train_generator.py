import argparse
import json
import logging
import os
import ruamel.yaml as yaml
import clip
from pathlib import Path

import numpy as np
import random
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.distributed as dist

# Dataset
from dataset import create_dataset, create_sampler, create_loader, normalize_fn
from models.model_gan_generator import NetG
import utils
   

def train(generator, model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    
    # generator train
    generator.train()
    
    # train
    model.eval()  
    
    loss_image = nn.CrossEntropyLoss()
    loss_text = nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 100
    step_size = 100
    warmup_iterations = warmup_steps*step_size 

    scaler = GradScaler()

    for batch_idx,(image, text, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        batch_size = len(image)
        image = image.to(device,non_blocking=True)   
        idx = idx.to(device,non_blocking=True)   
        text = tokenizer(text, truncate=True).to(device)
        
        # encode text to get text embedding
        text_embedding = model.module.encode_text(text)
        sec_emb = text_embedding
        
        # generate noise 
        batch_size = image.size(0)
        noise = torch.randn(batch_size, 100).to(device)
        gen_image, _ = generator(noise, sec_emb)
        delta_im = gen_image
        
        # limit the perturbation to a range of [-epsilon, epsilon]
        norm_type = args.norm_type
        epsilon = args.epsilon
        if norm_type == "l2":
            temp = torch.norm(delta_im.view(delta_im.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            delta_im = delta_im * epsilon / temp
        elif norm_type == "linf":
            delta_im = torch.clamp(delta_im, -epsilon / 255., epsilon / 255)  # torch.Size([16, 3, 256, 256])

        delta_im = delta_im.to(image.device)
        delta_im = F.interpolate(delta_im, (image.shape[-2], image.shape[-1]))
        
        # add delta(noise) to image
        image_adv = torch.clamp(image + delta_im, min=0, max=1)
        # normalize the image_adv
        image_adv_norm = normalize_fn(image_adv)

        # use image_adv_norm as input
        image_input = image_adv_norm
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        logits_per_image, logits_per_caption= model(image_input, text)                  
        ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)
        total_loss = (loss_image(logits_per_image, ground_truth) + loss_text(logits_per_caption, ground_truth)) / 2
        
        loss = -total_loss

        loss.backward()
        optimizer.step()  
        
        # gather loss
        reduced_loss = total_loss.clone()
        dist.reduce(reduced_loss, 0)  # average across GPUs

        if dist.get_rank() == 0:  # print loss
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {reduced_loss.item()}')
        
        metric_logger.update(total_loss=reduced_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch==0 and batch_idx%step_size==0 and batch_idx<=warmup_iterations: 
            scheduler.step(batch_idx//step_size)  
               
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logging.info(f"Averaged stats: {metric_logger.global_avg()}")     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


def main(args, config):
    if args.distributed:
        utils.init_distributed_mode(args) 
    
    if utils.is_main_process():
        log_level = logging.INFO
        utils.setup_logging(os.path.join(config['output_dir'], "out.log"), log_level)  
    
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    generator = NetG()
    # checkpoint = torch.load("/remote-home/songtianwei/research/unlearn_multimodal/output/gen_flickr_ViT-B_16/checkpoint_epoch_50.pth", map_location=device)
    # generator.load_state_dict(checkpoint['model'])
    generator = generator.to(device)
    
    optimizerG = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=10, gamma=0.1)
    
    clip_model = args.clip_model
    logging.info("Creating attacked model - CLIP {}".format(clip_model))
    model, _ = clip.load(clip_model, device, jit=False)
    model = model.float()
    model = model.to(device) 
    tokenizer = clip.tokenize
    
    # Distribute training
    model_without_ddp = model
    generator_without_ddp = generator
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[args.gpu])
        model_without_ddp = model.module   
        generator_without_ddp = generator.module

    #### Dataset #### 
    logging.info("Creating clean dataset for {}".format(config['dataset']))
    train_dataset, test_dataset, val_dataset = create_dataset('train_generator', config)
    logging.info(f"Training dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}, Testing dataset size:{len(test_dataset)}")  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers, batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2, num_workers=[4,4,4], is_trains=[True, False, False],collate_fns=[None,None,None])   

    # start training
    import time 
    cur_time = time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time()))
    max_epoch = config['schedular']['epochs']
    generator_save_name = 'direct_generator_'+cur_time+'.pth'
    for epoch in range(max_epoch):
        logging.info(f"Start training epoch {epoch+1}/{max_epoch}")
        
        train_stats = train(generator, model, train_loader, optimizerG, tokenizer, epoch, warmup_steps=0, device=device, scheduler=schedulerG, config=config)
        
        save_obj = {
            'model': generator_without_ddp.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'lr_scheduler': schedulerG.state_dict(),
            'config': config,
            'epoch': epoch,
        }
        torch.save(save_obj, os.path.join(config['output_dir'], f'checkpoint_epoch_{epoch+1}.pth'))  


if __name__ == '__main__':

    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/gen_Flickr.yaml')
    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--clip_model', default='ViT-B/16', type=str)
    
    # noise limit
    parser.add_argument('--norm_type', default='l2', type=str, choices=['l2', 'linf'])
    parser.add_argument('--epsilon', default=8, type=int, help='perturbation')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action="store_true")
    
    args = parser.parse_args()
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    clip_model_str = args.clip_model.replace('/', '_')
    output_dir = "./output/temp_gen_{}_{}".format(config['dataset'], clip_model_str)
    config.update({'output_dir': output_dir})
    
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(config["output_dir"], 'config.yaml'), 'w')) 

    main(args, config)