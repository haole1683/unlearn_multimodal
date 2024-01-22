import torch
from torchvision import transforms

import clip

from utils.data_utils import load_class_dataset
from utils.noise_utils import gen_perturbation
from utils.clip_util import prompt_templates
from utils.noise_utils import save_noise

from models.model_gan_generator import NetG
from tqdm import tqdm

myTrans = transforms.Compose([
    # transforms.Resize((224,224)),
    transforms.ToTensor()
])
device = "cuda:0"

def generate_noise_from_pretrain(generator_path, clip_version='RN50'):
    trainDataset, testDataset = load_class_dataset('CIFAR10',myTrans)

    model, _ = clip.load(clip_version, device, jit=False)
    model = model.float()
    model = model.to(device) 
    
    tokenizer = clip.tokenize
    generator = NetG()
    generator = generator.to(device)

    generator.load_state_dict(torch.load(generator_path))
    generator.eval()

    text_prompts = [prompt.format("cat") for prompt in prompt_templates]

    text_tokens = tokenizer(text_prompts).to(device)
    text_embedding = model.encode_text(text_tokens)

    noise_shape = (32,3,32,32)
    noise_list = []
    
    noise_count = 5000

    for i in tqdm(range(noise_count//noise_shape[0] + 1)):
        delta_im = gen_perturbation(generator, text_embedding[0], noise_shape, args=None)
        noise_list.append(delta_im)

    noise = torch.concat(noise_list)
    noise = noise[:noise_count]
 
    clip_version = clip_version.replace('/','-')
    tgt_save_path = "/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn/cat_noise_{}.pt".format(clip_version)

    torch.save(delta_im.detach().cpu(), tgt_save_path)