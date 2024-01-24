import torch
from torchvision import transforms

import clip

from utils.data_utils import load_class_dataset
from utils.noise_utils import gen_perturbation
from utils.clip_util import prompt_templates
from utils.noise_utils import save_noise
from utils.data_utils import jsonDataset

from torch.utils.data import Dataset, DataLoader

from models.model_gan_generator import NetG
from tqdm import tqdm

myTrans = transforms.Compose([
    # transforms.Resize((224,224)),
    transforms.ToTensor()
])
device = "cuda:0"


def generate_noise_from_pretrain(generator_path, clip_version='RN50'):
    # Generate noise for cifar-10 Cat class
    
    # trainDataset, testDataset = load_class_dataset('CIFAR10',myTrans)

    model, _ = clip.load(clip_version, device, jit=False)
    model = model.float()
    model = model.to(device) 
    tokenizer = clip.tokenize
    
    if clip_version == 'RN50':
        text_embedding_dim = 1024
    elif clip_version == 'ViT-B/32':
        text_embedding_dim = 512
    else:
        text_embedding_dim = 512    
    clip_version = clip_version.replace('/','-')
    
    
    generator = NetG(ngf=text_embedding_dim//8)
    generator = generator.to(device)

    generator.load_state_dict(torch.load(generator_path))
    generator.eval()
    
    
    def gen1():
        print("Generating noise for cat class in cifar-10 cat class [3,32,32] image")

        text_prompts = [prompt.format("cat") for prompt in prompt_templates]

        text_tokens = tokenizer(text_prompts).to(device)
        text_embedding = model.encode_text(text_tokens)

        noise_shape = (16,3,32,32)
        noise_list = []
        
        noise_count = 5000

        for i in tqdm(range(noise_count//noise_shape[0] + 1)):
            rand_idx = torch.randint(0,10000,(1,)).item() % len(text_embedding)
            with torch.no_grad():
                delta_im = gen_perturbation(generator, text_embedding[rand_idx], noise_shape,evaluate=True, args=None)
            noise_list.append(delta_im)

        noise1 = torch.concat(noise_list)
        noise1 = noise1[:noise_count]
    
        tgt_save_path = "/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn/cat_noise_{}.pt".format(clip_version)
        torch.save(noise1.detach().cpu(), tgt_save_path)
    
    
    def gen2():
        # Generator noise for Cat class dataset image-pair dataset
        print("Generating noise for cat class in cat class image-pair dataset [3,224,224] image")
        gen2 = True
        if not gen2:
            return
        
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
        
        dataset_len = len(catDs)
        noises2 = torch.rand(dataset_len,3,224,224).to(device)
        noise_shape = (16,3,224,224)
        
        for batch in tqdm(myDataloader):
            text, img, index = batch
            text = tokenizer(text, truncate=True).to(device)
            text_embedding = model.encode_text(text)
            with torch.no_grad():
                delta_im = gen_perturbation(generator, text_embedding, noise_shape,evaluate=True, args=None)
            noises2[index] = delta_im

        tgt_save_path = "/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn/cat_noise_ori_{}.pt".format(clip_version)
        torch.save(noises2.detach().cpu(), tgt_save_path)    

generator_path = "/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn/generator_versionRN50_epoch200_loss0.11304377764463425.pth"
generate_noise_from_pretrain(generator_path)