import argparse
import json
import logging
import os
import ruamel.yaml as yaml
from pathlib import Path

import numpy as np
import random
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist

# Dataset
from dataset import create_dataset, create_sampler, create_loader, normalize_fn
from models.model_gan_generator import NetG
import utils
import clip

def generate_poison_dataset(generator, model, data_loader, tokenizer, device, config, args):
    
    # generator eval
    generator.eval()
    
    # model eval
    model.eval()  
    
    loss_image = nn.CrossEntropyLoss()
    loss_text = nn.CrossEntropyLoss()
    
    dataset_name = config['dataset']
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Generate Poison Dataset from dataset {}'.format(dataset_name)
    print_freq = 100
    step_size = 100
    json_dict_list = []
    image_idx = 1
    batch_idx = 1
    poison_data_save_path = os.path.join(config['output_dir'], "poison_data")
    Path(poison_data_save_path).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx,(image, text, idx, anno) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            batch_size = len(image)
            image = image.to(device,non_blocking=True)   
            idx = idx.to(device,non_blocking=True)   
            text_ids = tokenizer(text, truncate=True).to(device)
            # text = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(device) 
            
            # encode text to get text embedding
            if args.distributed:
                text_embedding = model.module.encode_text(text_ids)
            else:
                text_embedding = model.encode_text(text_ids)
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
            
            # For debug
            # image_paths = anno
            # image_paths = [path.split('/')[-1].split('.')[0] for path in image_paths]
            # save_names = [str(image_idx) + ".pt" for image_idx in image_paths]
            # delta_im = torch.stack([torch.load(os.path.join(poison_data_save_path, save_names[t]))['noise'] for t in range(len(save_names))])
            # delta_im = delta_im.to(image.device)
            
            delta_im = F.interpolate(delta_im, (image.shape[-2], image.shape[-1]))
            # add delta(noise) to image
            image_adv = torch.clamp(image + delta_im, min=0, max=1)
            # normalize the image_adv
            image_adv_norm = normalize_fn(image_adv)

            # use image_adv_norm as input
            image_input = image_adv_norm
            
            logits_per_image, logits_per_caption= model(image_input, text_ids)                  
            ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)
            total_loss = (loss_image(logits_per_image, ground_truth) + loss_text(logits_per_caption, ground_truth)) / 2
            
            # gather loss
            if args.distributed:
                reduced_loss = total_loss.clone()
                dist.reduce(reduced_loss, 0)  # 使用reduce将损失从所有卡汇总到主卡（0号卡）

                if dist.get_rank() == 0:  # 只有主卡打印损失
                    print(f'Generate, Distributed Batch {batch_idx}, Loss: {reduced_loss.item()}')
                metric_logger.update(total_loss=reduced_loss.item())
            else:
                print(f'Generate, Singal Batch {batch_idx}, Loss: {total_loss.item()}')
                metric_logger.update(total_loss=total_loss.item())
                        
            for i in range(batch_size):
                image_path = anno[i]
                image_idx = image_path.split('/')[-1].split('.')[0] 
                save_name = str(image_idx) + ".pt"
                
                data_dict = {
                    # save the delta(noise) as image instead of image
                    # "image_adv":image_adv[i].detach().cpu(),
                    "noise":delta_im[i].detach().cpu(),
                }
                tensor_save_path = os.path.join(poison_data_save_path, save_name)
                if not args.debug:
                    torch.save(data_dict, tensor_save_path)
                json_dict = {}
                json_dict['image'] = save_name 
                json_dict['caption'] = text[i]
                json_dict['image_id'] = idx[i].detach().cpu().numpy().tolist()
                json_dict_list.append(json_dict)
            batch_idx += 1 
            
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logging.info(f"Averaged stats: {metric_logger.global_avg()}")     

    json_save_folder = "./data/poison/"
    Path(json_save_folder).mkdir(parents=True, exist_ok=True)
    json_save_path = os.path.join(json_save_folder, f"poison_{dataset_name}_clip.json")
    json_data = json.dumps(json_dict_list)
    # write to json file
    with open(json_save_path, 'w') as f:
        f.write(json_data)
    logging.info(f"Save json file to {json_save_path}")
    logging.info(f"Generate poison dataset done!")

   
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
    if args.checkpoint is None or not os.path.exists(args.checkpoint):
        logging.info("No checkpoint provided. Please run train_generator first.")
        raise "No checkpoint provided"
    else:
        logging.info("loading from " + args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        generator.load_state_dict(checkpoint['model'])
    generator.to(device)
    
    #### CLIP Model ####   
    logging.info("Creating attacked model - CLIP {}".format(config['clip_model']))
    model, _ = clip.load(config['clip_model'], device, jit=False)
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
    train_dataset, test_dataset, val_dataset = create_dataset('re_gen_dataset', config)
    logging.info(f"Training dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}, Testing dataset size:{len(test_dataset)}")  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers, batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2, num_workers=[4,4,4], is_trains=[False, False, False],collate_fns=[None,None,None])   

    generate_poison_dataset(generator, model, train_loader, tokenizer, device, config, args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/Flickr_poison.yaml')
    
    parser.add_argument('--checkpoint', default='/remote-home/songtianwei/research/unlearn_multimodal/output/cifar10-Pretrain/checkpoint_epoch_65.pth')   
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action="store_true")
    parser.add_argument('--debug', action="store_true")
    
    # noise limit
    parser.add_argument('--norm_type', default='l2', type=str, choices=['l2', 'linf'])
    parser.add_argument('--epsilon', default=8, type=int, help='perturbation')
    
    args = parser.parse_args()
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config = {**config['common'], **config['step2']}

    config['output_dir'] = os.path.join(config['poison_delta_root'], config['dataset'])
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)

    main(args, config)