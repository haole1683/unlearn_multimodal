import torch
from torchvision import transforms

import clip

from utils.data_utils import load_class_dataset, To244TensorTrans
from utils.noise_utils import gen_perturbation
from utils.clip_util import prompt_templates
from utils.noise_utils import save_noise, limit_noise
from utils.data_utils import jsonDataset
from utils.os_utils import add_index_to_json_file, create_folder

from torch.utils.data import Dataset, DataLoader

from models.model_gan_generator import NetG
from tqdm import tqdm
import os
import argparse

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
        
    def get_prompt(self, label_name=None, update=True):
        prompt = self.prompt_list[self.cur_index]
        if update:
            self.update_prompt()
        if label_name is not None:
            prompt = prompt.format(label_name)
        return prompt
    
    def get_embedding(self):
        self.update_prompt()
        return self.text_embedding[self.cur_index]
    
    
        
def generate_noise_from_pretrain(args):
    # Generate noise for cifar-10 Cat class
    clip_model = args.clip_model
    if clip_model == 'both':
        text_embedding_dim = 512
        # model, _ = clip.load("RN50", args.device, jit=False)   
        model, _ = clip.load(clip_model, args.device, jit=False)
        model = model.float()
        model = model.to(args.device)  
    else:
        model, _ = clip.load(clip_model, args.device, jit=False)
        model = model.float()
        model = model.to(args.device)  
    if clip_model == 'RN50':
        text_embedding_dim = 1024
    elif clip_model == 'ViT-B/32':
        text_embedding_dim = 512
    elif clip_model == 'RN101':
        text_embedding_dim = 512
    elif clip_model == 'RN50x4':
        text_embedding_dim = 640
    elif clip_model == 'ViT-B/16':
        text_embedding_dim = 512
    else:
        raise ValueError("Invalid clip model")
    
    clip_model = clip_model.replace('/','-')
    
    generator = NetG(ngf=text_embedding_dim//8)
    generator = generator.to(args.device)
    generator_path = args.generator_path
    checkpoint = torch.load(generator_path)
    
    if 'module.' in list(checkpoint.keys())[0]:
        generator.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
    else:
        generator.load_state_dict(checkpoint)
    generator.eval()
    
    def gen1():
        # Here noise is over random, so here's a strategy
        z_latent_strategy = zLatentStrategy(update_freq=args.update_z_freq)
        the_tgt_class = args.tgt_class
        prompt_Strategy = promptStrategy(strategy=args.text_prompt_stragegy, tokenizer=clip.tokenize, text_encoder=model, label_word=the_tgt_class)
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
        
        # the_fixed_noise = torch.randn(noise_shape).to(args.device)
        # the_fixed_noise = limit_noise(the_fixed_noise, epsilon=128)
        for i in tqdm(range(noise_count//noise_shape[0] + 1)):
            z_lantent = z_latent_strategy.getZLatent(noise_shape[0])
            print("The text prompt is ", prompt_Strategy.get_prompt(label_name=the_tgt_class ,update=False))
            text_embedding = prompt_Strategy.get_embedding()
            with torch.no_grad():
                delta_im = gen_perturbation(generator, text_embedding, noise_shape, 
                                            z_latent=z_lantent,evaluate=True, args=args)
            noise_list.append(delta_im)
            # delta_im = the_fixed_noise
        noise1 = torch.concat(noise_list)
        noise1 = noise1[:noise_count]
    
        noise_shape_str = str(noise_count) + "-" + '-'.join([str(i) for i in noise_shape[1:]])
        the_gen_1_output_dir = os.path.join(args.output_dir, args.dataset)
        create_folder(the_gen_1_output_dir)
        tgt_save_path = os.path.join(the_gen_1_output_dir, f"noise_gen1_{noise_shape_str}_{clip_model}_{the_tgt_class}.pt")
        torch.save(noise1.detach().cpu(), tgt_save_path)
        
        return noise1.detach().cpu()
    
    tokenizer = clip.tokenize
    def gen2():
        # Here noise is over random, so here's a strategy
        z_latent_strategy = zLatentStrategy(update_freq=args.update_z_freq)
        # Generator noise for tgt_class class dataset image-pair dataset
        tgt_class = args.tgt_class
        print(f"Generating noise for {tgt_class} class in {tgt_class} class image-pair dataset [3,224,224] image")
        
        json_tgt_path = f"./data/laion-{tgt_class}-with-index.json"
            
        if not os.path.exists(json_tgt_path):
            # add index for the json file
            json_tgt_no_index_path = f"./data/laion-{tgt_class}.json"
            add_index_to_json_file(json_tgt_no_index_path)

        tgtClassDs = jsonDataset(json_tgt_path, img_transform = To244TensorTrans, contain_index=True)
        myDataloader = DataLoader(tgtClassDs, batch_size=16, shuffle=True,drop_last=True)

        dataset_len = len(tgtClassDs)
        print(f"Total {dataset_len} images in {tgt_class} class dataset")
        noises2 = torch.rand(dataset_len,3,224,224)
        # noises2 = torch.rand(dataset_len,3,288,288)
        noise_shape = (3,224,224)
        # noise_shape = (16,3,288,288)
        
        for batch in tqdm(myDataloader):
            img, text, index = batch
            text = tokenizer(text, truncate=True).to(args.device)
            text_embedding = model.encode_text(text)
            z_latent = z_latent_strategy.getZLatent(text.shape[0])
            with torch.no_grad():
                delta_im = gen_perturbation(generator, text_embedding, noise_shape,
                                            z_latent,evaluate=True, args=args)
            index = index % dataset_len
            noises2[index] = delta_im.detach().cpu()
            
        noise_count = len(tgtClassDs)
        noise_shape_str = str(noise_count) + "-" + '-'.join([str(i) for i in noise_shape[1:]])
        tgt_save_path = os.path.join(args.output_dir, f"noise_gen2_{noise_shape_str}_{tgt_class}_{clip_model}.pt")
        torch.save(noises2, tgt_save_path)    
    
    # For gen1
    if args.gen_which == 'gen1' or args.gen_which == 'all':
        origin_dataset = args.dataset
        origin_tgt_class = args.tgt_class
        if origin_dataset == 'all':
            dataset_list = ['cifar10', 'stl10']
        else:
            dataset_list = [origin_dataset]
        for dataset in dataset_list:
            args.dataset = dataset
            if origin_tgt_class == 'all':
                if args.dataset == 'cifar10':
                    tgt_class_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                elif args.dataset == 'stl10':
                    tgt_class_list = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
            else:
                tgt_class_list = [origin_tgt_class]
            noise_dict = {}
            for tgt_class in tgt_class_list:
                print(f"Generate noise for {tgt_class} class")
                args.tgt_class = tgt_class
                the_tgt_noise = gen1()
                noise_dict[tgt_class] = the_tgt_noise
            all_save_path = os.path.join(args.output_dir,args.dataset, f"noise_gen1_{clip_model}_{args.dataset}_all.pt")
            torch.save(noise_dict, os.path.join(all_save_path))
        args.dataset = origin_dataset
        args.tgt_class = origin_tgt_class
    
    # For gen2
    if args.gen_which == 'gen2' or args.gen_which == 'all':
        gen2()

def main(args):
    generator_path = args.generator_path
    generator_clip_version = generator_path.split('/')[-3].split('-')[-1]
    
    if generator_clip_version != args.clip_model:
        print(f"!!!!!!!!!!!Generator clip version {generator_clip_version} is not equal to args.clip_model {args.clip_model}!!!!!!!!!!!!")
    clip_model_str = "-encoder-" + args.clip_model.replace('/','-')
    if generator_clip_version == 'both':
        args.output_dir = os.path.join(args.output_dir, generator_clip_version + clip_model_str)
    else:
        args.output_dir = os.path.join(args.output_dir , generator_clip_version)
    if args.overwrite:
        if os.path.exists(args.output_dir):
                os.system("rm -rf {}".format(args.output_dir)) 
    create_folder(args.output_dir)
    generate_noise_from_pretrain(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()       
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--generator_path', default= "./output/unlearn_stage1_train_g_unlearn/gen_all-both/checkpoint/generator_best_epoch-214_loss-0.11523310208746033.pth")
    parser.add_argument('--output_dir', default="./output/unlearn_stage2_generate_noise/")
    
    parser.add_argument('--clip_model', default='RN101', help="image encoder type of clip", choices=['RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'ViT-B/16', 'both'])
    parser.add_argument('--dataset', default='all', choices=['all', 'cifar10', 'stl10', 'imagenet-cifar10'])
    parser.add_argument('--tgt_class', default='all', choices=['all', 'cat', 'dog', 'bird', 'car', 'truck', 'plane', 'ship', 'horse', 'deer'])
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--gen_which', default='all', choices=['gen1', 'gen2', 'all'])
    # generate noise hyper parameter
    # Here, z freq means the frequency of updating z (generator latent input)
    # if 1 , then every batch update z_input
    parser.add_argument('--update_z_freq', default=10000000, type=int, help="Update z frequency")
    # strategy for text prompt, random, fixed, poll
    # total 20 template for text prompt
    parser.add_argument('--text_prompt_stragegy', default='fixed', choices=['random', 'fixed', 'poll'])
    
    parser.add_argument('--noise_shape', default=(3,224,224), type=tuple)
    args = parser.parse_args()

    main(args)