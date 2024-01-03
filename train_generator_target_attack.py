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

from models.model_gan_generator import NetG
from utils.ori_utils import MetricLogger, SmoothedValue
from utils.nce import InfoNCE
from utils.data_utils import load_dataset, create_sampler, create_loader, get_dataset_class
from utils.patch_utils import de_normalize
from utils.distributed_utils import init_distributed_mode, is_main_process, setup_logging, get_rank, get_world_size
from utils.clip_util import clip_normalize
from utils.noise_utils import gen_perturbation
# from utils.metrics import KL, CE, umap

def train(generator, model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, target_dataloader=None):
    
    # generator train
    generator.train()
    
    # train
    model.eval()  
    
    # InfoNCE loss 
    # This contrastive loss enforces the embeddings of similar (positive) samples to be close
    # and those of different (negative) samples to be distant.
    criterion_contrastive = InfoNCE()
    similarity_loss = nn.CosineSimilarity()
    
    loss_image = nn.CrossEntropyLoss()
    loss_text = nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('total_loss', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 100
    step_size = 100
    warmup_iterations = warmup_steps*step_size 

    scaler = GradScaler()
    
    if args.distributed:
        the_model = model.module
    else:
        the_model = model

    guide_text = get_dataset_class(args.attack_dataset)
    prompt = "An image of a {}" 
    guide_text_prompt = [prompt.format(class_name) for class_name in guide_text]
    guide_text_token = tokenizer(guide_text_prompt, truncate=True).to(device)
    
    for batch_idx, (image, text, labels, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        batch_size = len(image)
        image = image.squeeze(1) .to(device,non_blocking=True)   
        text = text.squeeze().to(device, non_blocking=True)
        
        text_adv_tokens = guide_text_token[train_idx % len(guide_text_token)].unsqueeze(0).repeat(batch_size, 1)
        # 用 advCLIP 导入数据集对应的text已经是token了，不需要再tokenizer
        
        image = de_normalize(image)
        
        # encode text to get text embedding
        text_embedding = the_model.encode_text(text_adv_tokens)
        sec_emb = text_embedding
        
        # generate noise 
        batch_size = image.size(0)
        noise = torch.randn(batch_size, 100).to(device)
        
        delta_im = gen_perturbation(generator, text_embedding, image, args)
        
        # add delta(noise) to image
        image_adv = torch.clamp(image + delta_im, min=0, max=1)
        # normalize the image_adv
        image_norm = clip_normalize(image)
        image_adv_norm = clip_normalize(image_adv)

        # zero the parameter gradients
        optimizer.zero_grad()
            
        image_clean_emb = the_model.encode_image(image_norm.to(device)) 
        image_adv_emb = the_model.encode_image(image_adv_norm.to(device))
        
        text_clean_emb = the_model.encode_text(text.to(device))
        text_adv_emb = the_model.encode_text(text_adv_tokens.squeeze().to(device))
        
        # image and guide text embedding are closer
        loss_advImg_advText = criterion_contrastive(image_adv_emb, text_adv_emb).mean()   # min the loss
        
        loss_advImg_advText_cosSim = similarity_loss(image_adv_emb, text_adv_emb).mean()
        # image embedding with noise are closer
        # image_adv_emb_flip = image_adv_emb.flip(0)
        # loss_advImg_advImg = criterion_contrastive(image_adv_emb, image_adv_emb_flip).mean()  # min the loss
        
        adv_loss1 = -loss_advImg_advText_cosSim
        # adv_loss2 = loss_text_advImg

        adv_loss = args.alpha * adv_loss1 
        # adv_loss = args.beta * adv_loss1 +  loss_advImg_advImg
        
        # umap_loss_pos1 = - umap(clean_image_emb, adv_image_emb)
        # umap_loss_pos2 = - umap(adv_image_emb, clean_text_emb)
        # umap_loss = umap_loss_pos1 + args.gamma * umap_loss_pos2

        # G_loss = args.alpha * adv_loss + args.delta * umap_loss
        G_loss = adv_loss
        
        loss = G_loss
        loss.backward()
        optimizer.step()  
        
        train_idx += 1
        
        # gather loss
        reduced_loss = loss.clone()
        if args.distributed:
            dist.reduce(reduced_loss, 0)  # average across GPUs
        if args.distributed and dist.get_rank() == 0:  # print loss
            print(f'Train_idx {train_idx}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')
        elif not args.distributed:
            print(f'Train_idx {train_idx}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')
        
        metric_logger.update(total_loss=reduced_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch==0 and batch_idx%step_size==0 and batch_idx<=warmup_iterations: 
            scheduler.step(batch_idx//step_size)  
             
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logging.info(f"Averaged stats: {metric_logger.global_avg()}")     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


def main(args):
    if args.distributed:
        init_distributed_mode(args) 
    
    if is_main_process():
        log_level = logging.INFO
        setup_logging(os.path.join(args.output_dir, "out.log"), log_level)  
    
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    generator = NetG()
    # init weights
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

    logging.info("Creating clean dataset for {}".format(args.dataset))
    dataset = load_dataset(args.dataset, args.batch_size)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    
    if args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
        
    train_loader, test_loader = create_loader([train_dataset, test_dataset],samplers, batch_size=[args.batch_size, args.batch_size], num_workers=[4,4], is_trains=[True, False],collate_fns=[None,None])
    
    # start training
    import time 
    cur_time = time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time()))
    max_epoch = args.max_epoch
    generator_save_name = 'direct_generator_'+cur_time+'.pth'
    for epoch in range(max_epoch):
        logging.info(f"Start training epoch {epoch+1}/{max_epoch}")
        
        train_stats = train(generator, model, train_loader, optimizerG, tokenizer, epoch, warmup_steps=0, device=device, scheduler=schedulerG)
        
        save_obj = {
            'model': generator_without_ddp.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'lr_scheduler': schedulerG.state_dict(),
            'epoch': epoch,
            'clip_loss': train_stats['total_loss'],
        }
        if epoch % 5 ==0:  # save model every 5 epochs
            torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))  


if __name__ == '__main__':

    parser = argparse.ArgumentParser()     
    # model
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--clip_model', default='ViT-B/16', type=str)
    
    # dataset
    parser.add_argument('--dataset', default='nus-wide', type=str, choices=['Flickr', 'COCO',
                                                                            'nus-wide', 'pascal', 
                                                                            'wikipedia', 'xmedianet'])
    parser.add_argument('--attack_dataset', default='CIFAR100', type=str, choices=['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet', 'STL10', 'GTSRB'])
    parser.add_argument('--batch_size', default=16, type=int)
    
    # noise limit
    parser.add_argument('--norm_type', default='l2', type=str, choices=['l2', 'linf'])
    parser.add_argument('--epsilon', default=8, type=int, help='perturbation')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action="store_true")
    
    # hyperparameters
    parser.add_argument('--alpha', default=10, type=float, help='weight for the adversarial loss')
    parser.add_argument('--beta', default=5, type=float, help='weight for the adversarial loss')
    parser.add_argument('--gamma', default=5, type=float, help='weight for the adversarial loss')
    parser.add_argument('--delta', type=int, default=1)
    parser.add_argument('--max_epoch', default=20, type=int)
    
    # output
    # parser.add_argument('--output_dir', default='./output', type=str)
    args = parser.parse_args()

    clip_model_str = args.clip_model.replace('/', '-')
    
    output_dir = "./output/text_targeted_gen_{}_{}_{}".format(args.dataset,args.attack_dataset,clip_model_str)
    args.output_dir = output_dir
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)