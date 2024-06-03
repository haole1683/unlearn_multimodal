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
from torchvision.utils import save_image
json_cat_path = "/remote-home/songtianwei/research/unlearn_multimodal/data/laion-cat-with-index.json"

myTrans = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

catDs = jsonDataset(json_cat_path, img_transform = myTrans)

myDataloader = DataLoader(catDs, batch_size=8, shuffle=False,drop_last=True)

device = "cuda:0"

clip_version = 'RN50'
model, _ = clip.load(clip_version, device, jit=False)
model = model.float()
model = model.to(device) 

text_embedding_dim = model.text_projection.shape[1]
generator = NetG(ngf=text_embedding_dim//8)
generator = generator.to(device)


generator_path = "/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn/generator_versionRN50_epoch200_loss0.11304377764463425.pth"
generator.load_state_dict(torch.load(generator_path))

generator.eval()

freeze_encoder = model.visual
logging.info("freeze image encoder")
for param in freeze_encoder.parameters():
    param.requires_grad = False

freeze_encoder = model.transformer
logging.info("freeze text encoder")
for param in freeze_encoder.parameters():
    param.requires_grad = False

model.eval()
generator.eval()

tokenizer = clip.tokenize

loss_image = nn.CrossEntropyLoss()
loss_text = nn.CrossEntropyLoss()
infoNCE_loss = InfoNCE()

loop = tqdm(myDataloader, desc='Train')
the_noises = [torch.ones(3,224,224)] * len(catDs) 
visualize = False

tgt_path = "/remote-home/songtianwei/research/unlearn_multimodal/useless_code"

with torch.no_grad():
    for batch in loop:
        text = tokenizer(batch[0], truncate=True).to(device)
        imgs = batch[1].to(device)
        index = batch[2]
        batch_size = imgs.shape[0]
        
        text_embedding = model.encode_text(text)
        delta_im = gen_perturbation(generator, text_embedding, imgs.shape, args=None)
        
        print(delta_im[0].eq(delta_im[1]))
        
        
        
        for i in range(batch_size):
            the_noises[index[i]] = delta_im[i]
        
        images_adv = torch.clamp(imgs + delta_im, min=0, max=1)
        if visualize:
            save_image(imgs, "text_img_clean.jpg")
            save_image(images_adv, "test_img_adv.jpg")
            print(imgs.eq(images_adv))
        images_adv = clip_normalize(images_adv)
        
        unlearn_img_embedding = model.encode_image(images_adv)
        
        loss = infoNCE_loss(unlearn_img_embedding, text_embedding)
        
        the_loss_value = loss.detach().cpu().numpy()
        
        loop.set_postfix(loss = the_loss_value)
        break
    noise_name = "cat_noise_{}".format(clip_version.replace("/","-"))
    save_tgt = "/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn/cat_noise_ori.pt"
    torch.save(the_noises, save_tgt)