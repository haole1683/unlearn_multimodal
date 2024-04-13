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
from utils import distributed_utils

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel
import json

from tqdm import tqdm
import logging
import time

import logging


class jsonRecord:
    def __init__(self, path):
        self.data = {}
        self.path = path
        
    def add(self, key, value):
        self.data[key] = value
        
    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.data, f)
            
    def save_args(self, args):
        self.data['args'] = vars(args)
        self.save()
    
    def save_exp_res(self, exp_res : dict):
        if 'experiment_result' not in self.data:
            self.data['experiment_result'] = []
        self.data['experiment_result'].append(exp_res)
        self.save()

def train(epoch_idx, train_dataloader, clip_models, generator, optimizerG, 
          schedulerG, tokenizer,
          myJsonRecord, args):
    
    device = args.device
    clip_version = args.clip_model.replace("/", "_")
    
    loss_image = nn.CrossEntropyLoss()
    loss_text = nn.CrossEntropyLoss()
    infoNCE_loss = InfoNCE()
    
    loss_list = []
    
    output_dir = args.output_dir
    g_save_path = os.path.join(output_dir, "checkpoint")
    
    loop = tqdm(train_dataloader, desc='Train')
    batch_total = len(train_dataloader)
    for batch_idx, batch in enumerate(loop):
        imgs = batch[0].to(device)
        
        if args.img_transform == 'kornia':
            imgs = augmentations_kornia(imgs)
        
        text = tokenizer(batch[1], truncate=True).to(device)
        index = batch[2]
        batch_size = imgs.shape[0]
        
        losses_of_models = []
        
        # cal the both loss of double version clip
        for model_idx in range(len(clip_models)):
            clip_model = clip_models[model_idx]
            text_embeddings = clip_model.encode_text(text)
            delta_im = gen_perturbation(generator, text_embeddings, imgs.shape, args=args)
            
            images_adv = torch.clamp(imgs + delta_im, min=0, max=1)
            
            image_clean = clip_normalize(imgs)
            images_adv = clip_normalize(images_adv)
            
            img_embeddings_clean = clip_model.encode_image(image_clean)
            img_embeddings_unlearn = clip_model.encode_image(images_adv)
            text_embeddings = clip_model.encode_text(text)
            
            # Method1 to calculate loss
            loss_contrastive_imgs = infoNCE_loss(img_embeddings_unlearn, img_embeddings_clean)
            loss_contrastive_unlearn_text = infoNCE_loss(img_embeddings_unlearn, text_embeddings)
            # NOTE : other_imgs is the negative samples...which is not defined
            # negetive_img_embedding = clip_model.encode_image(other_imgs)
            
            # Method2 to calculate loss (adv_feature, text_feature)
            logits_per_image, logits_per_caption= clip_model(images_adv, text)                  
            ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)
            loss_contrastive_img_text = (loss_image(logits_per_image, ground_truth) + loss_text(logits_per_caption, ground_truth)) / 2
            
            alpha, beta, gamma = 1, 1, 1
            total_loss = loss_contrastive_imgs * alpha + loss_contrastive_unlearn_text * beta + loss_contrastive_img_text * gamma
            losses_of_models.append(total_loss)
        
        loss = sum(losses_of_models) / len(losses_of_models)
    
        the_loss_value = loss.detach().cpu().numpy()
        loss_list.append(the_loss_value)
        
        loss.backward()
        optimizerG.step()
        optimizerG.zero_grad()
        # schedulerG.step()
        
        loop.set_description(f'Epoch[{epoch_idx}]- Batch [{batch_idx+1}/{batch_total}]')
        loop.set_postfix(loss = the_loss_value)
    mean_loss = np.mean(loss_list)
    logging.info("epoch {} loss: {}".format(epoch_idx, mean_loss))
    record_dict = {
        "epoch": epoch_idx,
        "loss": float(mean_loss)
    }
    myJsonRecord.save_exp_res(record_dict)
    
    # save the cur generator model 
    if epoch_idx % 10 == 0:
        torch.save(generator.state_dict(), os.path.join(g_save_path, "generator_all_version{}_epoch{}_loss{}.pth".format(clip_version,epoch_idx, mean_loss)))
        

def process_clip_model(clip_model, device):
    clip_model = clip_model.float()
    clip_model = clip_model.to(device)
    
    # NOTE Freeze the visual and text encoder
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
    
    # fix the seed for reproducibility
    seed = args.seed + distributed_utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    # dataset
    if args.trainset == 'all':
        json_path = "/remote-home/songtianwei/research/unlearn_multimodal/data/laion_cifar10.json"
        args.output_dir = os.path.join(args.output_dir, "gen_all")
    elif args.trainset == 'cat':
        json_path = "/remote-home/songtianwei/research/unlearn_multimodal/data/laion-cat-with-index-ttt.json"
        args.output_dir = os.path.join(args.output_dir, "gen_cat")
    
    # logging
    if args.overwrite:
        if os.path.exists(args.output_dir):
            os.system("rm -rf {}".format(args.output_dir))
    Path(os.path.join(args.output_dir, "log")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, "checkpoint")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, "json")).mkdir(parents=True, exist_ok=True)
    
    cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    clip_version = args.clip_model
    clip_version = clip_version.replace("/", "_")
    log_tgt_path = os.path.join(args.output_dir, "log/log_all_generator_{}.txt".format(clip_version))
    print(log_tgt_path)
    
    logging.basicConfig(filename=log_tgt_path, level=logging.DEBUG, 
                        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s')
    
    myJsonRecord = jsonRecord(os.path.join(args.output_dir, "json/exp_record.json"))
    myJsonRecord.save_args(args)
    
    myTrans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
        
    trainDataset = jsonDataset(json_path, img_transform = myTrans, contain_index=True)
    trainDataloader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True,drop_last=True)

    device = args.device

    # clip
    clip_version = args.clip_model
    if clip_version == 'both':
        clip_model_resnet,_ = clip.load("RN101", device, jit=False)
        clip_model_vit,_ = clip.load("ViT-B/16", device, jit=False)
        text_embedding_dim_resnet = clip_model_resnet.text_projection.shape[1]
        text_embedding_dim_vit = clip_model_vit.text_projection.shape[1]
        if text_embedding_dim_resnet != text_embedding_dim_vit:
            print(text_embedding_dim_resnet, text_embedding_dim_vit)
            raise ValueError("text embedding dim not equal")
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
    # update the optimizer lr from 0.0001 -> 0.1
    optimizerG = torch.optim.Adam(generator.parameters(), lr=0.1, betas=(0.0, 0.9))
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=10, gamma=0.1)
    
    epoch = args.epoch
    
    logging.info("Start training")
    
    if args.distributed:
        distributed_utils.init_distributed_mode(args)  
        clip_models_ddp = [DistributedDataParallel(clip_model, device_ids=[args.gpu]) for clip_model in clip_models]
        clip_models = [clip_model.module for clip_model in clip_models_ddp]
        
        
    if args.distributed and distributed_utils.is_main_process():
        # TODO modify the log path
        log_level = logging.INFO
        log_name = f"finetune_clip_dataset-{args.finetune_dataset}_{'poison' if args.poisoned else 'natural'}.log"
        distributed_utils.setup_logging(os.path.join(args.output_dir, log_name), log_level)

    for epoch_idx in range(epoch):
        train(epoch_idx, trainDataloader, clip_models, generator, optimizerG, schedulerG, tokenizer, myJsonRecord, args)
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()       
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action="store_true")
    parser.add_argument('--finetune_dataset', default='myLaion')
    
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    
    parser.add_argument('--trainset', default='all', choices=['all', 'cat'])

    # poisoning
    parser.add_argument('--clip_model', default='RN50', help="image encoder type of clip", choices=['RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'ViT-B/16', 'both'])
    # parser.add_argument('--freeze_encoder', default='', help="image or text or none") # fi/ft = freeze image/text

    # transform for image
    parser.add_argument('--img_transform', default='kornia', choices=['None', 'kornia'])

    parser.add_argument('--output_dir', default='/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage1_train_g_unlearn')
    parser.add_argument('--overwrite', action='store_true')
    
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # log_testing()

    main(args)
