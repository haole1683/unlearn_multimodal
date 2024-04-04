import argparse
import json
import logging
import os
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
from utils.metrics_utils import InfoNCE
from utils.data_utils import (
    load_dataset, jsonDataset,
    augmentations_kornia
)
from utils.patch_utils import de_normalize
from utils.noise_utils import gen_perturbation
from utils.clip_util import _convert_image_to_rgb, clip_transform, clip_normalize, prompt_templates, zeroshot_classifier

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import json
from PIL import Image

from tqdm import tqdm
import logging
import time

def train(epoch_idx, train_dataloader, clip_models, generator, optimizerG, 
          schedulerG, tokenizer, g_save_path,
          args):
    
    device = args.device
    clip_version = args.clip_model.replace("/", "_")
    
    loss_image = nn.CrossEntropyLoss()
    loss_text = nn.CrossEntropyLoss()
    infoNCE_loss = InfoNCE()
    
    loss_list = []
    loss_sum = 0
    
    loop = tqdm(train_dataloader, desc='Train')
    batch_total = len(train_dataloader)
    for batch_idx, batch in enumerate(loop):
        imgs = batch[0].to(device)
        
        if args.img_transform == 'kornia':
            imgs = augmentations_kornia(imgs)
        
        text = tokenizer(batch[1], truncate=True).to(device)
        index = batch[2]
        batch_size = imgs.shape[0]
        
        losses = []
        for model_idx in range(len(clip_models)):
            clip_model = clip_models[model_idx]
            text_embeddings = clip_model.encode_text(text)
            delta_im = gen_perturbation(generator, text_embeddings, imgs.shape, args=None)
            
            images_adv = torch.clamp(imgs + delta_im, min=0, max=1)
            images_adv = clip_normalize(images_adv)
            
            # unlearn_img_embeddings = [clip_model.encode_image(images_adv) for clip_model in clip_models]
            # loss = infoNCE_loss(unlearn_img_embedding, text_embedding)
            
            logits_per_image, logits_per_caption= clip_model(images_adv, text)                  
            ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)
            total_loss = (loss_image(logits_per_image, ground_truth) + loss_text(logits_per_caption, ground_truth)) / 2
            loss = total_loss
            
            losses.append(loss)
        
        if len(losses) > 1:
            # average 
            loss = sum(losses) / len(losses)
        else:
            loss = losses[0]
        
        the_loss_value = loss.detach().cpu().numpy()
        loss_list.append(the_loss_value)
        
        loss.backward()
        optimizerG.step()
        optimizerG.zero_grad()
        schedulerG.step()
        
        loop.set_description(f'Epoch[{epoch_idx}]- Batch [{batch_idx+1}/{batch_total}]')
        loop.set_postfix(loss = the_loss_value)
    mean_loss = np.mean(loss_list)
    logging.info("epoch {} loss: {}".format(epoch_idx, mean_loss))
    # save the cur generator model 
    if epoch_idx % 20 == 0:
        torch.save(generator.state_dict(), os.path.join(g_save_path, "generator_all_version{}_epoch{}_loss{}.pth".format(clip_version,epoch_idx, mean_loss)))
        

def process_clip_model(clip_model, device):
    clip_model = clip_model.float()
    clip_model = clip_model.to(device)
    
    # NOTE 这里 Freeze住 CLIP的两个 encoder
    freeze_encoder = clip_model.visual
    logging.info("freeze image encoder")
    for param in freeze_encoder.parameters():
        param.requires_grad = False
    freeze_encoder = clip_model.transformer
    logging.info("freeze text encoder")
    for param in freeze_encoder.parameters():
        param.requires_grad = False
        
    clip_model.eval()
    return clip_model


def main(args):

    # dataset
    json_path = "/remote-home/songtianwei/research/unlearn_multimodal/data/laion_cifar10.json"    

    myTrans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
        
    trainDataset = jsonDataset(json_path, img_transform = myTrans, contain_index=True)
    trainDataloader = DataLoader(trainDataset, batch_size=8, shuffle=True,drop_last=True)

    device = args.device

    # clip
    clip_version = args.clip_model
    if clip_version == 'both':
        clip_model_resnet,_ = clip.load("RN50", device, jit=False)
        clip_model_vit,_ = clip.load("RN50", device, jit=False)
        clip_models = [clip_model_resnet, clip_model_vit]
    else:
        clip_model, _ = clip.load(clip_version, device, jit=False)
        clip_models = [clip_model]
    clip_models = [process_clip_model(clip_model, device) for clip_model in clip_models]
    
    # tokenizer
    tokenizer = clip.tokenize
    
    # generator
    text_embedding_dim = clip_models[0].text_projection.shape[1]
    generator = NetG(ngf=text_embedding_dim//8)
    generator = generator.to(device)
    generator.train()

    # optimizer
    # update the optimizer lr from 0.0001 -> 0.001
    optimizerG = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.0, 0.9))
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=10, gamma=0.1)
    
    epoch = 201
    clip_version = clip_version.replace("/", "_")

    output_dir = os.path.join(args.output_dir, "gen_all")
    cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_tgt_path = os.path.join(output_dir, "log/log_all_generator_{}.txt".format(clip_version))
    logging.basicConfig(filename=log_tgt_path, level=logging.INFO)
    g_save_path = os.path.join(output_dir, "checkpoint")
    
    Path(g_save_path).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, "log")).mkdir(parents=True, exist_ok=True)
    
    logging.info("Start training")

    for epoch_idx in range(epoch):
        train(epoch_idx, trainDataloader, clip_models, generator, optimizerG, schedulerG, tokenizer, g_save_path, args)
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()       
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action="store_true")
    parser.add_argument('--finetune_dataset', default='myLaion')

    # poisoning
    parser.add_argument('--clip_model', default='both', help="image encoder type of clip", choices=['RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'ViT-B/16', 'both'])
    parser.add_argument('--freeze_encoder', default='', help="image or text or none") # fi/ft = freeze image/text

    # transform for image
    parser.add_argument('--img_transform', default='kornia', choices=['None', 'kornia'])

    parser.add_argument('--output_dir', default='/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
