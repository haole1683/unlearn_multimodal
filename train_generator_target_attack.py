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
import utils.ori_utils as utils 
from utils.nce import InfoNCE
from utils.load_data import load_dataset
from utils.patch_utils import de_normalize

def KL(P,Q,mask=None):
    eps = 0.0000001
    d = (P+eps).log()-(Q+eps).log()
    d = P*d
    if mask !=None:
        d = d*mask
    return torch.sum(d)
def CE(P,Q,mask=None):
    return KL(P,Q,mask)+KL(1-P,1-Q,mask)

def umap(output_net, target_net, eps=0.0000001):
    # Normalize each vector by its norm
    (n, d) = output_net.shape
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0
    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0
    # Calculate the cosine similarity
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
    model_distance = 1-model_similarity #[0,2]
    model_distance[range(n), range(n)] = 3
    model_distance = model_distance - torch.min(model_distance, dim=1)[0].view(-1, 1)
    model_distance[range(n), range(n)] = 0
    model_similarity = 1-model_distance
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))
    target_distance = 1-target_similarity
    target_distance[range(n), range(n)] = 3
    target_distance = target_distance - torch.min(target_distance,dim=1)[0].view(-1,1)
    target_distance[range(n), range(n)] = 0
    target_similarity = 1 - target_distance
    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0
    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)
    # Calculate the KL-divergence
    loss = CE(target_similarity,model_similarity)
    return loss


def train(generator, model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, target_dataloader=None):
    
    # generator train
    generator.train()
    
    # train
    model.eval()  
    
    criterion_contrastive = InfoNCE()
    
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

    guide_text = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    prompt = "An image of a {}"
    guide_text_prompt = [prompt.format(class_name) for class_name in guide_text]
    guide_text_token = tokenizer(guide_text_prompt, truncate=True).to(device)
    
    for batch_idx, (image, text, labels, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        batch_size = len(image)
        image = image.to(device,non_blocking=True)   
        image = image.squeeze(1)
        
        text = text.squeeze().to(device, non_blocking=True)
        # idx = idx.to(device,non_blocking=True)   
        # text = tokenizer(text, truncate=True).to(device)
        
        # 用 advCLIP 导入数据集对应的text已经是token了，不需要再tokenizer
        
        need_denorm = True
        if need_denorm:
            image = de_normalize(image)
        
        # encode text to get text embedding
        if args.distributed:
            text_embedding = model.module.encode_text(text)
        else:
            text_embedding = model.encode_text(text)
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
        image_norm = normalize_fn(image)
        image_adv_norm = normalize_fn(image_adv)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        if args.distributed:
            the_model = model.module
        else:
            the_model = model
            
        image_clean_emb = the_model.encode_image(image_norm.to(device)) 
        image_adv_emb = the_model.encode_image(image_adv_norm.to(device))
        
        text_clean_emb = the_model.encode_text(text.to(device))
        text_adv_emb = the_model.encode_text(guide_text_token.squeeze().to(device))
        
        loss_advImg_advText = criterion_contrastive(image_adv_emb, text_adv_emb).mean()   # min the loss
        # loss_text_advImg = criterion_contrastive(adv_image_emb, guide_text_emb).mean()  # max the loss
        
        adv_loss1 = loss_advImg_advText
        # adv_loss2 = loss_text_advImg

        adv_loss = args.alpha * adv_loss1 
        # adv_loss = args.beta * adv_loss1 +  adv_loss2
        
        # umap_loss_pos1 = - umap(clean_image_emb, adv_image_emb)
        # umap_loss_pos2 = - umap(adv_image_emb, clean_text_emb)
        # umap_loss = umap_loss_pos1 + args.gamma * umap_loss_pos2

        # G_loss = args.alpha * adv_loss + args.delta * umap_loss
        G_loss = adv_loss
        
        loss = G_loss

        loss.backward()
        
        optimizer.step()  
        
        # gather loss
        reduced_loss = loss.clone()

        if args.distributed:
            dist.reduce(reduced_loss, 0)  # average across GPUs

        if args.distributed and dist.get_rank() == 0:  # print loss
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {adv_loss.item()}')
        
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
        utils.init_distributed_mode(args) 
    
    if utils.is_main_process():
        log_level = logging.INFO
        utils.setup_logging(os.path.join(args.output_dir, "out.log"), log_level)  
    
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    generator = NetG()
    # init weights
    nn.init.kaiming_normal_(generator.weight.data)
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
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
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
    parser.add_argument('--max_epoch', default=200, type=int)
    
    # output
    # parser.add_argument('--output_dir', default='./output', type=str)
    args = parser.parse_args()

    clip_model_str = args.clip_model.replace('/', '-')
    
    output_dir = "./output/text_targeted_gen_{}_{}".format(args.dataset, clip_model_str)
    args.output_dir = output_dir
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)