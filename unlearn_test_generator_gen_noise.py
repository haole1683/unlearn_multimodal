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

class zLatentStrategy():
    def __init__(self, update_freq=100) -> None:
        self.z_latent = None
        self.total_sample_count = 0
        self.update_time = 0
        self.update_freq = update_freq
    
    def update_z(self, batch_size=None):
        if batch_size != None:
            self.z_latent = torch.randn(batch_size,100).to(args.device)
    
    def getZLatent(self, batch_size=None):
        self.update(batch_size)
        return self.z_latent
    
    def update(self, batch_size):
        self.total_sample_count += batch_size
        if self.z_latent == None:
            self.z_latent =  torch.randn(batch_size,100).to(args.device)
            return 
        elif self.total_sample_count / self.update_freq > self.update_time:
            self.update_time += 1
            self.update_z(batch_size)
            print(f"Update z latent at {self.total_sample_count} samples")
        else:
            return self.z_latent
            
class promptStrategy():
    def __init__(self, tokenizer, text_encoder, label_word='cat', strategy="fix") -> None:
        self.strategy = strategy
        self.prompt_list = prompt_templates
        self.text_prompts = []
        self.text_tokens = []
        self.text_embedding = []
        self.cur_index = 0
        
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = self.text_encoder.text_projection.device
        
        self.label_word = label_word
        
        self.init_token_embedding()
    
    def init_token_embedding(self):
        self.text_prompts = [prompt.format(self.label_word) for prompt in prompt_templates]

        self.text_tokens = self.tokenizer(self.text_prompts).to(self.device)
        self.text_embedding = self.text_encoder.encode_text(self.text_tokens)
        
    def update_prompt(self):
        if self.strategy == "random":
            self.cur_index = torch.randint(0,len(self.prompt_list),(1,)).item()
        elif self.strategy == 'fix':
            self.cur_index = 0
        elif self.strategy == 'poll':
            self.cur_index = (self.cur_index + 1) % len(self.prompt_list)
        
    def get_prompt(self, label_name=None):
        prompt = self.prompt_list[self.cur_index]
        self.update_prompt()
        if label_name is not None:
            prompt = prompt.format(label_name)
        return prompt
    
    def get_embedding(self):
        self.update_prompt()
        return self.text_embedding[self.cur_index]
    
    
        
        
        
        
def generate_noise_from_pretrain(generator_path, clip_version='RN50'):
    # Generate noise for cifar-10 Cat class
    model, _ = clip.load(clip_version, args.device, jit=False)
    model = model.float()
    model = model.to(args.device) 
    
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
    
    tgt_class = 'cat'
    
    # Here noise is over random, so here's a strategy
    z_latent_strategy = zLatentStrategy(update_freq=args.update_z_freq)
    
    def gen1():
        prompt_Strategy = promptStrategy(strategy=args.text_prompt_stragegy, tokenizer=clip.tokenize, text_encoder=model)
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
        
        noise_list = []

        for i in tqdm(range(noise_count//noise_shape[0] + 1)):
            z_lantent = z_latent_strategy.getZLatent(noise_shape[0])
            text_embedding = prompt_Strategy.get_embedding()
            with torch.no_grad():
                delta_im = gen_perturbation(generator, text_embedding, noise_shape, 
                                            z_latent=z_lantent,evaluate=True, args=args)
            noise_list.append(delta_im)

        noise1 = torch.concat(noise_list)
        noise1 = noise1[:noise_count]
    
        noise_shape_str = str(noise_count) + "-" + '-'.join([str(i) for i in noise_shape[1:]])

        tgt_save_path = "/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn/noise_gen1_{}_{}_{}.pt".format(noise_shape_str,tgt_class,clip_version)
        torch.save(noise1.detach().cpu(), tgt_save_path)
    
    tokenizer = clip.tokenize
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
            z_lantent = z_latent_strategy.getZLatent(text.shape[0])
            with torch.no_grad():
                delta_im = gen_perturbation(generator, text_embedding, noise_shape,
                                            z_lantent,evaluate=True, args=args)
            index = index % dataset_len
            noises2[index] = delta_im
            
        noise_count = len(tgtClassDs)
        noise_shape_str = str(noise_count) + "-" + '-'.join([str(i) for i in noise_shape[1:]])
        tgt_save_path = "/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn/noise_gen2_{}_{}_{}.pt".format(noise_shape_str,tgt_class,clip_version)
        torch.save(noises2.detach().cpu(), tgt_save_path)    

    gen1()
    # gen2()

def main(args):
    generator_path = args.generator_path
    generate_noise_from_pretrain(generator_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()       
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'stl10', 'imagenet-cifar10'])
    parser.add_argument('--generator_path', default= "/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn/generator_versionRN50_epoch200_loss0.11304377764463425.pth")
    parser.add_argument('--output_dir', default="/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn/")
    
    # generate noise hyper parameter
    # Here, z freq means the frequency of updating z (generator latent input)
    # if 1 , then every batch update z_input
    parser.add_argument('--update_z_freq', default=1000, type=int, help="Update z frequency")
    # strategy for text prompt, random, fixed, poll
    # total 20 template for text prompt
    parser.add_argument('--text_prompt_stragegy', default='random', choices=['random', 'fixed', 'poll'])
    
    parser.add_argument('--noise_shape', default=(3,224,224), type=tuple)
    args = parser.parse_args()

    main(args)