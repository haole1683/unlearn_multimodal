import torch
from torchvision import transforms

import clip

from utils.data_utils import load_class_dataset
from utils.noise_utils import gen_perturbation
from utils.clip_util import prompt_templates
from utils.noise_utils import save_noise
from utils.data_utils import jsonDataset
from utils.os_utils import add_index_to_json_file

from torch.utils.data import Dataset, DataLoader

from models.model_gan_generator import NetG
from tqdm import tqdm
import os
import argparse

myTrans = transforms.Compose([
    # transforms.Resize((224,224)),
    transforms.ToTensor()
])



def generate_noise_from_pretrain(generator_path, clip_version='RN50'):
    # Generate noise for cifar-10 Cat class
    
    # trainDataset, testDataset = load_class_dataset('CIFAR10',myTrans)

    model, _ = clip.load(clip_version, args.device, jit=False)
    model = model.float()
    model = model.to(args.device) 
    tokenizer = clip.tokenize
    
    if clip_version == 'RN50':
        text_embedding_dim = 1024
    elif clip_version == 'ViT-B/32':
        text_embedding_dim = 512
    else:
        text_embedding_dim = 512    
    clip_version = clip_version.replace('/','-')
    
    
    generator = NetG(ngf=text_embedding_dim//8)
    generator = generator.to(args.device)

    generator.load_state_dict(torch.load(generator_path))
    generator.eval()
    
    tgt_class = 'truck'
    def gen1():
        test_dataset = args.dataset
        
        if test_dataset == 'cifar10':
            gen_batch = 32
            noise_shape = (gen_batch,3,32,32) # for cifar-10
            noise_count = 5000
        elif test_dataset == 'stl10':
            gen_batch = 16
            noise_shape = (gen_batch,3,96,96)
            noise_count = 500
        
        print(f"Generating noise for tgt class in {args.dataset} tgt class {noise_shape} image")
        
        text_prompts = [prompt.format(tgt_class) for prompt in prompt_templates]

        text_tokens = tokenizer(text_prompts).to(args.device)
        text_embedding = model.encode_text(text_tokens)

        noise_list = []
        
        

        for i in tqdm(range(noise_count//noise_shape[0] + 1)):
            rand_idx = torch.randint(0,10000,(1,)).item() % len(text_embedding)
            with torch.no_grad():
                delta_im = gen_perturbation(generator, text_embedding[rand_idx], noise_shape,evaluate=True, args=args)
            noise_list.append(delta_im)

        noise1 = torch.concat(noise_list)
        noise1 = noise1[:noise_count]
    
        noise_shape_str = str(noise_count) + "-" + '-'.join([str(i) for i in noise_shape[1:]])

        tgt_save_path = "/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn/noise_gen1_{}_{}_{}.pt".format(noise_shape_str,tgt_class,clip_version)
        torch.save(noise1.detach().cpu(), tgt_save_path)
    
    
    def gen2():
        # Generator noise for tgt_class class dataset image-pair dataset
        print(f"Generating noise for {tgt_class} class in {tgt_class} class image-pair dataset [3,224,224] image")
        
        json_tgt_path = f"/remote-home/songtianwei/research/unlearn_multimodal/data/laion-{tgt_class}-with-index.json"
            
        if not os.path.exists(json_tgt_path):
            # add index for the json file
            json_tgt_no_index_path = f"/remote-home/songtianwei/research/unlearn_multimodal/data/laion-{tgt_class}.json"
            add_index_to_json_file(json_tgt_no_index_path)
            
        myTrans = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        tgtClassDs = jsonDataset(json_tgt_path, img_transform = myTrans, contain_index=True)
        myDataloader = DataLoader(tgtClassDs, batch_size=16, shuffle=True,drop_last=True)

        dataset_len = len(tgtClassDs)
        noises2 = torch.rand(dataset_len,3,224,224).to(args.device)
        noise_shape = (16,3,224,224)
        
        for batch in tqdm(myDataloader):
            img, text, index = batch
            text = tokenizer(text, truncate=True).to(args.device)
            text_embedding = model.encode_text(text)
            with torch.no_grad():
                delta_im = gen_perturbation(generator, text_embedding, noise_shape,evaluate=True, args=args)
            index = index % dataset_len
            noises2[index] = delta_im
            
        noise_count = len(tgtClassDs)
        noise_shape_str = str(noise_count) + "-" + '-'.join([str(i) for i in noise_shape[1:]])
        tgt_save_path = "/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn/noise_gen2_{}_{}_{}.pt".format(noise_shape_str,tgt_class,clip_version)
        torch.save(noises2.detach().cpu(), tgt_save_path)    

    gen1()
    gen2()

def main(args):
    generator_path = args.generator_path
    generate_noise_from_pretrain(generator_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()       
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'stl10', 'imagenet-cifar10'])
    parser.add_argument('--generator_path', default= "/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn/generator_versionRN50_epoch200_loss0.11304377764463425.pth")
    parser.add_argument('--output_dir', default="/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn/")
    args = parser.parse_args()

    main(args)