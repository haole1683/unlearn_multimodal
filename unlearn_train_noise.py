from utils.toolbox import PerturbationTool


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
from utils.metrics_utils import InfoNCE
from utils.data_utils import load_dataset
from utils.patch_utils import de_normalize
from utils.noise_utils import gen_perturbation
from utils.clip_util import _convert_image_to_rgb, clip_transform, clip_normalize, prompt_templates, zeroshot_classifier


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import json
from PIL import Image

from tqdm import tqdm
import logging

json_cat_path = "/remote-home/songtianwei/research/unlearn_multimodal/data/laion-cat.json"
json_nocat_path = "/remote-home/songtianwei/research/unlearn_multimodal/data/laion-no-cat.json"

class myDataset(Dataset):
    def __init__(self,json_path,text_transform=None, img_transform=None):
        self.json_data = json.load(open(json_path,'r'))
        self.text_transform = text_transform
        self.img_transform = img_transform
    def __getitem__(self,idx):
        sample = self.json_data[idx]
        text, img_path = sample['caption'], sample['image_path']
        img = Image.open(img_path)
        img = img.convert('RGB')
        if self.text_transform:
            text = self.text_transform(text)
        if self.img_transform:
            img = self.img_transform(img)
        return text, img
    def __len__(self):
        return len(self.json_data)
    
myTrans = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

catDs = myDataset(json_cat_path, img_transform = myTrans)
otherDs = myDataset(json_nocat_path, img_transform = myTrans)

myDataloader = DataLoader(catDs, batch_size=64, shuffle=False,drop_last=False)
otherDataloader = DataLoader(otherDs, batch_size=64, shuffle=True,drop_last=True)

device = "cuda:1"

from tqdm import tqdm


other_data_iter = iter(otherDataloader)
condition = True
train_idx = 0

clip_version = 'RN50'
model, _ = clip.load(clip_version, device, jit=False)
model = model.float()
model = model.to(device) 

# freeze the clip model
freeze_encoder = model.visual
logging.info("freeze image encoder")
for param in freeze_encoder.parameters():
    param.requires_grad = False
freeze_encoder = model.transformer
logging.info("freeze text encoder")
for param in freeze_encoder.parameters():
    param.requires_grad = False
model.eval()

clean_train_loader = myDataloader
base_model = model
# criterion = torch.nn.CrossEntropyLoss()
infoNCE_criterion = InfoNCE()
# optimizer = torch.optim.SGD(params=base_model.parameters(), lr=0.1, weight_decay=0.0005, momentum=0.9)
noise_generator = PerturbationTool(epsilon=0.03137254901960784, num_steps=20, step_size=0.0031372549019607846)  
sample_nums = len(clean_train_loader.dataset)

load_noise_path = "/remote-home/songtianwei/research/unlearn_multimodal/output/train_unlearn_noise/noise_epoch19_loss0.8868988156318665.pth"
if os.path.exists(load_noise_path):
    noise = torch.load(load_noise_path)["noise"]
else:
    noise = torch.zeros([sample_nums, 3, 224, 224])
    
epoch_num = 101
tokenizer = clip.tokenize

for epoch_idx in range(epoch_num):
    # optimize theta for M steps
    # base_model.train()
    # for param in base_model.parameters():
    #     param.requires_grad = True
    # for j in range(0, 10):
    #     try:
    #         (images, labels) = next(data_iter)
    #     except:
    #         train_idx = 0
    #         data_iter = iter(clean_train_loader)
    #         (images, labels) = next(data_iter)
        
    #     for i, _ in enumerate(images):
    #         # Update noise to images
    #         images[i] += noise[train_idx]
    #         train_idx += 1
    #     images, labels = images.cuda(), labels.cuda()
    #     base_model.zero_grad()
    #     optimizer.zero_grad()
    #     logits = base_model(images)
    #     loss = criterion(logits, labels)
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(base_model.parameters(), 5.0)
    #     optimizer.step()
    
    # Perturbation over entire dataset
    idx = 0
    for param in base_model.parameters():
        param.requires_grad = False
    
    loss_list = []
    loop = tqdm(enumerate(clean_train_loader), total=len(clean_train_loader))
    for i, (text, images) in loop:
        batch_start_idx, batch_noise = idx, []
        text = tokenizer(text, truncate=True)
        for i, _ in enumerate(images):
            # Update noise to images
            batch_noise.append(noise[idx])
            idx += 1
        batch_noise = torch.stack(batch_noise).to(device)
        
        # Update sample-wise perturbation
        base_model.eval()
        
        images, text = images.to(device), text.to(device)
        # if the other_data_iter is exhausted, reset it
        try:
            other_imgs = next(other_data_iter)[1].to(device)
        except:
            other_data_iter = iter(otherDataloader)
            other_imgs = next(other_data_iter)[1].to(device)
            
        loss_value, perturb_img, eta = noise_generator.min_min_CLIP_attack(device,images, text,other_imgs, base_model,clip_normalize, infoNCE_criterion, 
                                                          random_noise=batch_noise)
        for i, delta in enumerate(eta):
            noise[batch_start_idx+i] = delta.clone().detach().cpu()
        # loss_value = loss.detach().cpu().numpy()
        loss_list.append(loss_value)
        loop.set_description('Loss: %.4f' % (loss_value))
        
    # save the noise
    clip_version = clip_version.replace("/", "_")
    tgt_folder = "/remote-home/songtianwei/research/unlearn_multimodal/output/train_unlearn_noise_{}".format(clip_version)
    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)
    save_obj = {
        "noise": noise,
    }
    loss_avg = np.mean(loss_list)
    if epoch_idx % 10 == 0:
        torch.save(save_obj, os.path.join(tgt_folder, "noise_epoch{}_loss{}.pth".format(epoch_idx, loss_avg)))
    
    # Eval stop condition
    # eval_idx, total, correct = 0, 0, 0
    # for i, (images, labels) in enumerate(clean_train_loader):
    #     for i, _ in enumerate(images):
    #         # Update noise to images
    #         images[i] += noise[eval_idx]
    #         eval_idx += 1
    #     images, labels = images.cuda(), labels.cuda()
    #     with torch.no_grad():
    #         logits = base_model(images)
    #         _, predicted = torch.max(logits.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    # acc = correct / total
    # print('Accuracy %.2f' % (acc*100))
    # if acc > 0.99:
    #     condition=False      
