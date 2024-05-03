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
from utils.record_utils import setup_logging

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel
import json

from tqdm import tqdm
import logging
import time

# distrubute
from accelerate import Accelerator

# import torch
# NOTE : comment the line below when running
# torch.autograd.set_detect_anomaly(True)

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

def train(epoch_idx, accelerator, train_dataloader, clip_models, generator, optimizerG, 
          schedulerG, tokenizer,
          myJsonRecord, args):
    
    clip_version = args.clip_model.replace("/", "_")
    
    loss_image = nn.CrossEntropyLoss()
    loss_text = nn.CrossEntropyLoss()
    infoNCE_loss = InfoNCE()
    
    loss_list_dict = {}
    
    output_dir = args.output_dir
    g_save_path = os.path.join(output_dir, "checkpoint")
    
    if accelerator.is_main_process:
        loop = tqdm(train_dataloader, desc='Train')
    else:
        loop = train_dataloader
    batch_total = len(train_dataloader)

    for batch_idx, batch in enumerate(loop):
        imgs = batch[0]
        # TODO Test reverse image
        
        if args.img_transform == 'kornia':
            imgs = augmentations_kornia(imgs)
        
        text = tokenizer(batch[1], truncate=True)
        text = text.to(accelerator.device)
        index = batch[2]
        batch_size = imgs.shape[0]
        
        loss_dict = {}
        # cal the both loss of double version clip
        for model_idx in range(len(clip_models)):
            if hasattr(clip_models[model_idx], 'module'):
                clip_model = clip_models[model_idx].module
            else:
                clip_model = clip_models[model_idx]
            text_embeddings = clip_model.encode_text(text)
            delta_im = gen_perturbation(generator, text_embeddings, imgs.shape, args=args)
            # delta_im = torch.rand_like(imgs, requires_grad=True) * 0.1
            
            images_adv = torch.clamp(imgs + delta_im, min=0, max=1)
            
            image_clean = clip_normalize(imgs)
            images_adv = clip_normalize(images_adv)
            
            img_embeddings_clean = clip_model.encode_image(image_clean)
            img_embeddings_unlearn = clip_model.encode_image(images_adv)
            text_embeddings = clip_model.encode_text(text)
            
            # # Method1 to calculate loss
            loss_contrastive_imgs = infoNCE_loss(img_embeddings_unlearn, img_embeddings_clean)
            loss_contrastive_unlearn_text = infoNCE_loss(img_embeddings_unlearn, text_embeddings)
            # NOTE : other_imgs is the negative samples...which is not defined
            # negetive_img_embedding = clip_model.encode_image(other_imgs)
            
            # Method2 to calculate loss (adv_feature, text_feature)
            logits_per_image, logits_per_caption= clip_model(images_adv, text)                  
            ground_truth = torch.arange(batch_size, dtype=torch.long).to(accelerator.device)
            loss_contrastive_img_text = (loss_image(logits_per_image, ground_truth) + loss_text(logits_per_caption, ground_truth)) / 2
            
            alpha, beta, gamma = 1, 1, 1
            # total_loss = loss_contrastive_imgs * alpha + loss_contrastive_unlearn_text * beta + loss_contrastive_img_text * gamma
            total_loss = loss_contrastive_img_text * alpha
            # total_loss = loss_contrastive_img_text * alpha
            # total_loss = -loss_contrastive_unlearn_text * alpha
            
            # loss.backward()
            accelerator.backward(total_loss)
            
            loss_total_gather = accelerator.gather(total_loss).mean()
            loss_contrastive_img_text_gather = accelerator.gather(loss_contrastive_img_text).mean()
            loss_contrastive_imgs_gather = accelerator.gather(loss_contrastive_imgs).mean()
            loss_contrastive_unlearn_text_gather = accelerator.gather(loss_contrastive_unlearn_text).mean()
            
            optimizerG.step()
            optimizerG.zero_grad()
            
            if len(clip_models) == 1:
                model_key = args.clip_model
            elif len(clip_models) == 2 and model_idx == 0:
                model_key = "RN101"
            elif len(clip_models) == 2 and model_idx == 1:
                model_key = "ViT-B_16"
            else:
                raise ValueError("clip model not found")
            
            loss_dict[model_key] = {
                "lr": float(optimizerG.param_groups[0]['lr']),
                "loss": float(loss_total_gather.detach().cpu().numpy()),
                "loss_contrastive_imgs":  float(loss_contrastive_img_text_gather.detach().cpu().numpy()),
                "loss_contrastive_unlearn_text":  float(loss_contrastive_imgs_gather.detach().cpu().numpy()),
                "loss_contrastive_img_text":  float(loss_contrastive_unlearn_text_gather.detach().cpu().numpy())
            }
            
            # loss_dict[model_key +"_lr"] = optimizerG.param_groups[0]['lr']
            # loss_dict[model_key +"_loss"] = loss_total_gather.detach().cpu().numpy()
            # loss_dict[model_key +"_loss_contrastive_imgs"] = loss_contrastive_img_text_gather.detach().cpu().numpy()
            # loss_dict[model_key +"_loss_contrastive_unlearn_text"] = loss_contrastive_imgs_gather.detach().cpu().numpy()
            # loss_dict[model_key +"_loss_contrastive_img_text"] = loss_contrastive_unlearn_text_gather.detach().cpu().numpy()
        
        if accelerator.is_main_process:
            # one epoch loss average each model
            mean_lr_across_model = np.mean([loss_dict[model_key]['lr'] for model_key in loss_dict.keys()])
            mean_loss_across_model = np.mean([loss_dict[model_key]['loss'] for model_key in loss_dict.keys()])
            mean_loss_contrastive_images_across_model = np.mean([loss_dict[model_key]['loss_contrastive_imgs'] for model_key in loss_dict.keys()])
            mean_loss_contrastive_unlearn_text_across_model = np.mean([loss_dict[model_key]['loss_contrastive_unlearn_text'] for model_key in loss_dict.keys()])
            mean_loss_contrastive_img_text_across_model = np.mean([loss_dict[model_key]['loss_contrastive_img_text'] for model_key in loss_dict.keys()])
            
            loss_dict["mean_across_model"] = {
                "mean_lr_across_model": mean_lr_across_model,
                "mean_loss_across_model": mean_loss_across_model,
                "mean_loss_contrastive_images_across_model": mean_loss_contrastive_images_across_model,
                "mean_loss_contrastive_unlearn_text_across_model": mean_loss_contrastive_unlearn_text_across_model,
                "mean_loss_contrastive_img_text_across_model": mean_loss_contrastive_img_text_across_model
            }
            
            loss_list_dict[batch_idx] = loss_dict
            loop.set_description(f'Epoch[{epoch_idx}] - Batch [{batch_idx+1}/{batch_total}]')
            if args.clip_model == 'both':
                loss_rn, loss_vit = loss_dict["RN101"]["loss"], loss_dict["ViT-B_16"]["loss"]
                lr = optimizerG.param_groups[0]['lr']
                loop.set_postfix({"lr": lr,"loss":mean_loss_across_model, 'loss_rn':loss_rn, 'loss_vit':loss_vit})
            else:
                lr = optimizerG.param_groups[0]['lr']
                loop.set_postfix({"lr": lr,"loss":mean_loss_across_model})
    
    if accelerator.is_main_process:
        # all training process loss average
        train_mean_loss = np.mean([loss_dict["mean_across_model"]["mean_loss_across_model"] for loss_dict in loss_list_dict.values()])
        logging.info("epoch {} ,train_mean_loss: {}".format(epoch_idx, train_mean_loss))
        schedulerG.step()
        loss_mean = 114514
        if args.clip_model == 'both':
            loss_mean_rn = np.mean([loss_dict["RN101"]["loss"] for loss_dict in loss_list_dict.values()])
            loss_mean_vit = np.mean([loss_dict["ViT-B_16"]["loss"] for loss_dict in loss_list_dict.values()])
            loss_mean = np.mean([loss_mean_rn, loss_mean_vit])
            lr_mean = np.mean([loss_dict["RN101"]["lr"] for loss_dict in loss_list_dict.values()])
            record_dict = {
                "epoch": epoch_idx,
                "loss": loss_list_dict,
                "lr": lr_mean,
                "loss_1_avg": loss_mean_rn,
                "loss_2_avg": loss_mean_vit,
                "loss_avg": loss_mean
            }
        else:
            loss_mean = np.mean([loss_dict[args.clip_model]["loss"] for loss_dict in loss_list_dict.values()])
            lr_mean = np.mean([loss_dict[args.clip_model]["lr"] for loss_dict in loss_list_dict.values()])
            record_dict = {
                "epoch": epoch_idx,
                "loss": loss_list_dict,
                "lr": lr_mean,
                "loss_avg": loss_mean
            }
        myJsonRecord.save_exp_res(record_dict)

        # save the cur generator model 
        if epoch_idx % 20 == 0:
            torch.save(generator.state_dict(), os.path.join(g_save_path, "generator_all_version{}_epoch{}_loss{}.pth".format(clip_version,epoch_idx, train_mean_loss)))
        return loss_mean
    else:
        return 1145141919180

def process_clip_model(clip_model):
    clip_model = clip_model.float()
    # clip_model = clip_model.to(device)
    
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
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    # accelerator
    accelerator = Accelerator()
    print("here is the thread-idx", accelerator.local_process_index)
    
    # dataset
    if args.trainset == 'all':
        json_path = "/remote-home/songtianwei/research/unlearn_multimodal/data/laion_cifar10.json"
        clip_model_str = args.clip_model.replace("/", "_")
        args.output_dir = os.path.join(args.output_dir, "gen_all" + "-" + clip_model_str)
    elif args.trainset == 'cat':
        json_path = "/remote-home/songtianwei/research/unlearn_multimodal/data/laion-cat-with-index-ttt.json"
        clip_model_str = args.clip_model.replace("/", "_")
        args.output_dir = os.path.join(args.output_dir, "gen_cat" + "-" + clip_model_str)
        
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # logging
    if accelerator.is_main_process:
        if args.overwrite:
            if os.path.exists(args.output_dir):
                os.system("rm -rf {}".format(args.output_dir))
        Path(os.path.join(args.output_dir, "log")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(args.output_dir, "checkpoint")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(args.output_dir, "json")).mkdir(parents=True, exist_ok=True)
    
        cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        clip_version = args.clip_model
        clip_version = clip_version.replace("/", "_")
        log_tgt_path = os.path.join(args.output_dir, "log/log_all_generator_{}.log".format(clip_version))
        print(log_tgt_path)
        
        logging.basicConfig(filename=log_tgt_path, level=logging.DEBUG, 
                            format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s')
        
        myJsonRecord = jsonRecord(os.path.join(args.output_dir, "json/exp_record.json"))
        myJsonRecord.save_args(args)
        
    clip_version = args.clip_model
    myTrans224 = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    
    myTrans288 = transforms.Compose([
        transforms.Resize((288,288)),
        transforms.ToTensor()
    ])
    
    if clip_version == 'RN50x4':
        myTrans = myTrans288
    else:
        myTrans = myTrans224 
    trainDataset = jsonDataset(json_path, img_transform = myTrans, contain_index=True)
    trainDataloader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True,drop_last=True)

    # clip
    device = "cpu"
    if clip_version == 'both':
        clip_model_resnet,_ = clip.load("RN101", device, jit=False)
        clip_model_vit,_ = clip.load("ViT-B/16", device, jit=False)
        text_embedding_dim_resnet = clip_model_resnet.text_projection.shape[1]
        text_embedding_dim_vit = clip_model_vit.text_projection.shape[1]
        # "RN101" -> [512,512], "ViT-B/16" -> [512,512]
        # "RN50"  -> [512,1024],"ViT-B/32" -> [512,512]
        # "RN50x4"-> [640,640]
        if text_embedding_dim_resnet != text_embedding_dim_vit:
            print(text_embedding_dim_resnet, text_embedding_dim_vit)
            raise ValueError("text embedding dim not equal")
        clip_models = [clip_model_resnet, clip_model_vit]
    else:
        clip_model, _ = clip.load(clip_version, device, jit=False)
        clip_models = [clip_model]
    clip_models = [process_clip_model(clip_model) for clip_model in clip_models]
    
    # tokenizer
    tokenizer = clip.tokenize
    
    # generator
    text_embedding_dim = clip_models[0].text_projection.shape[1]
    generator = NetG(ngf=text_embedding_dim//8)
    # generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator)
    generator.train()

    # optimizer
    # update the optimizer lr from 0.0001 -> 0.01
    optimizerG = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
    # lr_scheduler - cosine anneling
    # schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=10, gamma=0.1)
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=args.epoch, eta_min=0.001)
    
    epoch = args.epoch
    
    logging.info("Start training")
    
    # move to accelerator
    trainDataloader = accelerator.prepare(trainDataloader)
    optimizerG = accelerator.prepare(optimizerG)
    clip_models = [accelerator.prepare(clip_model) for clip_model in clip_models]
    generator = accelerator.prepare(generator)
    
    if accelerator.is_main_process:
        log_level = logging.INFO
        log_name = f"finetune_clip_dataset-{args.finetune_dataset}.log"
        setup_logging(os.path.join(args.output_dir, log_name), log_level)

    epoch_min_loss = 1e3
    for epoch_idx in range(epoch):
        if accelerator.is_main_process:
            the_mean_loss = train(epoch_idx,accelerator, trainDataloader, clip_models, generator, optimizerG, schedulerG, tokenizer, myJsonRecord, args)
        else:
            the_mean_loss = train(epoch_idx,accelerator, trainDataloader, clip_models, generator, optimizerG, schedulerG, tokenizer, None, args)
            
        if the_mean_loss < epoch_min_loss:
            epoch_min_loss = the_mean_loss
            if accelerator.is_main_process:
                torch.save(generator.state_dict(), 
                           os.path.join(args.output_dir, "checkpoint/generator_best_epoch-{}_loss-{}.pth").format(epoch_idx, the_mean_loss))

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    parser = argparse.ArgumentParser()       
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--finetune_dataset', default='myLaion')
    
    parser.add_argument('--epoch', default=400, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    
    parser.add_argument('--trainset', default='all', choices=['all', 'cat'])

    # poisoning
    parser.add_argument('--clip_model', default='both', help="image encoder type of clip", choices=['RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'ViT-B/16', 'both', 'both2'])
    # parser.add_argument('--freeze_encoder', default='', help="image or text or none") # fi/ft = freeze image/text

    # transform for image
    parser.add_argument('--img_transform', default='kornia', choices=['None', 'kornia'])

    parser.add_argument('--output_dir', default='/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage1_train_g_unlearn')
    parser.add_argument('--overwrite', action='store_true')
    
    args = parser.parse_args()

    # log_testing()

    main(args)