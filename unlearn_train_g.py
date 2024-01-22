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
import utils.ori_utils as utils 
from utils.nce import InfoNCE
from utils.data_utils import load_dataset, jsonDataset
from utils.patch_utils import de_normalize
from utils.noise_utils import gen_perturbation
from utils.clip_util import _convert_image_to_rgb, clip_transform, clip_normalize, prompt_templates, zeroshot_classifier


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import json
from PIL import Image

from tqdm import tqdm
import logging

json_cat_path = "/remote-home/songtianwei/research/unlearn_multimodal/data/laion-cat-with-index.json"
json_nocat_path = "/remote-home/songtianwei/research/unlearn_multimodal/data/laion-no-cat.json"


    
myTrans = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

catDs = jsonDataset(json_cat_path, img_transform = myTrans)
otherDs = jsonDataset(json_nocat_path, img_transform = myTrans)

myDataloader = DataLoader(catDs, batch_size=16, shuffle=True,drop_last=True)
otherDataloader = DataLoader(otherDs, batch_size=16, shuffle=True,drop_last=True)

device = "cuda:0"

clip_version = 'RN50'
model, _ = clip.load(clip_version, device, jit=False)
model = model.float()
model = model.to(device) 

text_embedding_dim = model.text_projection.shape[1]
generator = NetG(ngf=text_embedding_dim//8)

generator_checkpoint_path = "/remote-home/songtianwei/research/unlearn_multimodal/output/train_unlearn_noise_RN50/noise_epoch90_loss0.07726054638624191.pth"

if generator_checkpoint_path is not None:
    generator.load_state_dict(torch.load(generator_checkpoint_path))
    logging.info("load generator checkpoint from {}".format(generator_checkpoint_path))
    
generator = generator.to(device)
generator.train()

# update the optimizer lr from 0.0001 -> 0.001
optimizerG = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.0, 0.9))
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=10, gamma=0.1)

freeze_encoder = model.visual
logging.info("freeze image encoder")
for param in freeze_encoder.parameters():
    param.requires_grad = False

freeze_encoder = model.transformer
logging.info("freeze text encoder")
for param in freeze_encoder.parameters():
    param.requires_grad = False

model.eval()

tokenizer = clip.tokenize

loss_image = nn.CrossEntropyLoss()
loss_text = nn.CrossEntropyLoss()
infoNCE_loss = InfoNCE()


clip_version = clip_version.replace('/','_')
loss_tgt_path = "/remote-home/songtianwei/research/unlearn_multimodal/record/losses/loss_{}.json".format(clip_version)


loss_list = []
loss_sum = 0
epoch = 201
clip_version = clip_version.replace("/", "_")
log_tgt_path = "/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn/log_{}.txt".format(clip_version)
logging.basicConfig(filename=log_tgt_path, level=logging.INFO)
g_save_path = "/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn"
logging.info("start training")


for epoch_idx in range(epoch):
    loop = tqdm(myDataloader, desc='Train')
    otherIter = iter(otherDataloader)
    for batch in loop:
        text = tokenizer(batch[0], truncate=True).to(device)
        imgs = batch[1].to(device)
        index = batch[2]
        batch_size = imgs.shape[0]
        
        noise = torch.randn(batch_size, 100).to(device)
        text_embedding = model.encode_text(text)
        delta_im = gen_perturbation(generator, text_embedding, imgs.shape, args=None)
        
        images_adv = torch.clamp(imgs + delta_im, min=0, max=1)
        images_adv = clip_normalize(images_adv)
        
        unlearn_img_embedding = model.encode_image(images_adv)
        
        # logits_per_image, logits_per_caption= model(images_adv, text)                  
        # ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)
        # total_loss = (loss_image(logits_per_image, ground_truth) + loss_text(logits_per_caption, ground_truth)) / 2
        # loss = total_loss
        other_imgs = next(otherIter)[1].to(device)
        other_imgs = clip_normalize(other_imgs)
        negetive_img_embedding = model.encode_image(other_imgs)
        loss = infoNCE_loss(unlearn_img_embedding, text_embedding, negetive_img_embedding)
        
        the_loss_value = loss.detach().cpu().numpy()
        loss_list.append(the_loss_value)
        loss.backward()
        optimizerG.step()
        optimizerG.zero_grad()
        
        loop.set_description(f'Epoch [{epoch+1}/{epoch_idx}]')
        loop.set_postfix(loss = the_loss_value)
    mean_loss = np.mean(loss_list)
    logging.info("epoch {} loss: {}".format(epoch_idx, mean_loss))
    # save the cur generator model 
    if epoch_idx % 20 == 0:
        torch.save(generator.state_dict(), os.path.join(g_save_path, "generator_version{}_epoch{}_loss{}.pth".format(clip_version,epoch_idx, mean_loss)))