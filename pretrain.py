import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
import logging
from pathlib import Path

import clip

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

def get_poisoned_similarity(args, config, model, device):
    dataset = create_dataset('poisoned_similariy', config)
    print("Test dataset size:", len(dataset))
    # print(dataset)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=4, drop_last=False)
    similarites = []
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for images, texts in dataloader:

        images = images.to(device)
        texts = clip.tokenize(texts, truncate=True, context_length=config['max_words']).to(device)
        with torch.no_grad():
            target_image_embeddings = model.encode_image(images)
            target_text_embeddings = model.encode_text(texts)
        sim = cos(target_image_embeddings, target_text_embeddings)
        similarites.append(torch.mean(sim).cpu())
    avg_sim = np.average(similarites) 
    print(f"Avg. Cosine Similarity of Poisoned Samples: {avg_sim:.4f}")
    result = {'avg_sim': avg_sim}
    return result

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, args):
    # train
    model.train()  
    
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

        optimizer.zero_grad()

        # with autocast():
        logits_per_image, logits_per_caption = model(image, text)
        ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)
        total_loss = (loss_image(logits_per_image, ground_truth) + loss_text(logits_per_caption, ground_truth)) / 2
        
        total_loss.backward()
        optimizer.step()
        
        # scaler.scale(total_loss).backward()
        # scaler.step(optimizer)
        # scaler.update()    
        
        # gather loss
        if args.distributed:
            reduced_loss = total_loss.clone()
            dist.reduce(reduced_loss, 0)  # 使用reduce将损失从所有卡汇总到主卡（0号卡）

            if dist.get_rank() == 0:  # 只有主卡打印损失
                logging.info(f'Pretrain,Distributed,Epoch {epoch},Batch {batch_idx},Loss: {reduced_loss.item()}')
            metric_logger.update(total_loss=reduced_loss.item())
        else:
            logging.info(f'Finetune,Singal,Epoch {epoch},Batch {batch_idx},Loss: {total_loss.item()}')
            metric_logger.update(total_loss=total_loss.item())

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch==0 and batch_idx%step_size==0 and batch_idx<=warmup_iterations: 
            scheduler.step(batch_idx//step_size)  
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logging.info(f"Averaged stats: {metric_logger.global_avg()}")     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  



@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval()  
    
    logging.info('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_embeds = []  
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = tokenizer(text, context_length=config['max_words']).to(device) 
        text_embed = model.encode_text(text_input)
        text_embed /= text_embed.norm(dim=-1, keepdim=True)
        text_embeds.append(text_embed)   
    text_embeds = torch.cat(text_embeds,dim=0)
    
    image_embeds = []
    for image, img_id in data_loader: 
        image = image.to(device) 
        image_embed = model.encode_image(image)        
        image_embed /= image_embed.norm(dim=-1, keepdim=True)     
        
        image_embeds.append(image_embed)
    
    image_embeds = torch.cat(image_embeds,dim=0)
    
    sims_matrix = image_embeds @ text_embeds.t()
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info('Evaluation time {}'.format(total_time_str)) 

    return sims_matrix.cpu().numpy(), sims_matrix.t().cpu().numpy()


            
@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result


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
    
    #### Model #### 
    logging.info("Creating model")
    model, _ = clip.load(config['clip_model'], device, jit=False)
    
    # set model parameters randn
    logging.info(f"Init model with {config['model_init']}")
    if config['model_init'] == 'randn':
        for p in model.parameters():
            p.data = torch.randn_like(p.data) * 0.01
    elif config['model_init'] == 'xavier':
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    elif config['model_init'] == 'kaiming':
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)
                
            
    
    model = model.float()
    model = model.to(device) 
    tokenizer = clip.tokenize

    start_epoch = 0

    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']+1
        logging.info(f'load checkpoint from {args.checkpoint}') 

    if args.freeze_encoder == 'image':
        freeze_encoder = model.visual
        logging.info("freeze image encoder")
        for param in freeze_encoder.parameters():
            param.requires_grad = False
    elif args.freeze_encoder == 'text':
        freeze_encoder = model.transformer
        logging.info("freeze text encoder")
        for param in freeze_encoder.parameters():
            param.requires_grad = False

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    #### Dataset #### 
    logging.info("Creating retrieval dataset for {}".format(config['dataset']))
    
    if args.poisoned:
        logging.info("Use poisoned dataset for finetune")
        poison_trainset, val_dataset, test_dataset = create_dataset('cifar-10-poison', config)
        train_dataset = poison_trainset
    else:
        logging.info("Use clean dataset for finetune")
        clean_trainset ,val_dataset, test_dataset = create_dataset("cifar-10", config)
        train_dataset = clean_trainset
    
    logging.info(f"Training dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}, Testing dataset size:{len(test_dataset)}")  
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers, batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2, num_workers=[4,4,4], is_trains=[True, False, False],collate_fns=[None,None,None])   


    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = start_epoch

    logging.info("Start training")
    start_time = time.time()    
    for epoch in range(start_epoch, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config, args)  
            
        # score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, tokenizer, device, config)
        # score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config)

        if utils.is_main_process():  
      
            # val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)  
            # logging.info(val_result)
            # test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)    
            # logging.info(test_result)

            # if args.poisoned:
            #     avg_sim = get_poisoned_similarity(args, config, model_without_ddp, device)
            #     logging.info(avg_sim)

            # if args.evaluate:   
            #     exp_type = 'fine-tuned'  if args.checkpoint else 'zero-shot' 
            #     if args.poisoned:
            #         avg_sim = get_poisoned_similarity(args, config, model_without_ddp, device)
            #         logging.info(avg_sim)
            #         log_stats = {
            #                     'avg_sim': '{}'.format(avg_sim['avg_sim']),
            #                     **{f'val_{k}': v for k, v in val_result.items()},
            #                     **{f'test_{k}': v for k, v in test_result.items()},
            #                     'epoch': epoch,
            #                     'time': datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            #                     'exp_type': exp_type,
            #                     }
            #     else:
            #         log_stats = {
            #                     **{f'val_{k}': v for k, v in val_result.items()},
            #                     **{f'test_{k}': v for k, v in test_result.items()},                  
            #                     'epoch': epoch,
            #                     'time': datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            #                     'exp_type': exp_type,
            #                     }
            #     with open(os.path.join(config['output_dir'], "evaluation_results.txt"),"a") as f:
            #         f.write(json.dumps(log_stats) + "\n")     
            # else:
            #     exp_type = 'fine-tuned'
            #     if args.poisoned:
            #         log_stats = {'avg_sim': '{}'.format(avg_sim['avg_sim']),
            #                     **{f'train_{k}': v for k, v in train_stats.items()},
            #                     **{f'val_{k}': v for k, v in val_result.items()},
            #                     **{f'test_{k}': v for k, v in test_result.items()},                  
            #                     'epoch': epoch,
            #                     'time': datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            #                     'exp_type': exp_type,
            #                     } 
            #     else:
            #         log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            #                     **{f'val_{k}': v for k, v in val_result.items()},
            #                     **{f'test_{k}': v for k, v in test_result.items()},                  
            #                     'epoch': epoch,
            #                     'time': datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            #                     'exp_type': exp_type,
            #                     }
            #     with open(os.path.join(config['output_dir'], "evaluation_results.txt"),"a") as f:
            #         f.write(json.dumps(log_stats) + "\n")   
                
                if (epoch + 1) % config['schedular']['save_freq'] == 0:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(config['output_dir'], f'checkpoint_epoch_{epoch+1}.pth'))  
                    logging.info(f"Save interval checkpoint of epoch {epoch+1} to {config['output_dir']}.")
                    
        if args.evaluate: 
            break
           
        lr_scheduler.step(epoch+warmup_steps+1)  
        if args.distributed:
            dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f'Training time {total_time_str}') 

    if utils.is_main_process():   
        with open(os.path.join(config['output_dir'], "evaluation_results.txt"),"a") as f:
            f.write("best epoch: %d\n"%best_epoch)               

            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/cifar10.yaml') 

    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action="store_true")

    # poisoning
    parser.add_argument('--poisoned', action='store_true', help="use poisoned dataset for training if not clean train")
    
    # model
    parser.add_argument('--freeze_encoder', default='', help="image or text or none") # fi/ft = freeze image/text

    # config overload
    parser.add_argument('--poisoned_ratio', default=1.0, type=float)
    
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config = {**config['common'], **config['Pretrain']}
    
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(config['output_dir'], 'config.yaml'), 'w')) 

    main(args, config)