import argparse
import torch
import clip
import os
from torchvision.datasets import  MNIST, CIFAR10, CIFAR100, ImageNet
import numpy as np
from tqdm import tqdm
import ruamel.yaml as yaml
import clip
from pathlib import Path

from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from models.model_gan_generator import NetG
import utils

import torch.nn.functional as F

# normalize from clip.py
def _convert_image_to_rgb(image):
    return image.convert("RGB")
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


# [224,224]
clip_transform = transforms.Compose([
    Resize((224,224),interpolation=BICUBIC),
    CenterCrop(224),
    _convert_image_to_rgb,
    ToTensor(),
])
clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

prompt_templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]


def zeroshot_classifier(clip_model, device, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = clip_model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def main(args):
    device = args.device
    
    # load clip model
    clip_model = args.clip_model
    model, preprocess = clip.load(clip_model, device)
    # make sure to convert the model parameters to fp32
    model = model.float()
    model = model.to(device) 
    
    # load generator
    checkpoint_path = args.checkpoint
    generator = NetG()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['model']).to(device)
    
    # load dataset
    dataset_name = args.dataset
    
    # zero-shot test dataset
    if dataset_name == 'MNIST':
        dataset = MNIST(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=clip_transform)
    elif dataset_name == 'CIFAR10':
        dataset = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=clip_transform)
    elif dataset_name == 'CIFAR100':
        dataset = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=clip_transform)
    elif dataset_name == 'ImageNet':
        dataset = ImageNet(root="/remote-home/songtianwei/research/unlearn_multimodal/data/imagenet", split='val', transform=clip_transform)
    else:
        raise NotImplementedError

    batch_size = args.batch_size
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    class_names = dataset.classes
    
    zeroshot_weights = zeroshot_classifier(clip_model, device, class_names, prompt_templates)
    
    # test
    test_method = args.test_method
    if test_method == 'clean':
        clean_test = True
        use_random = False
    elif test_method == 'adversarial':
        clean_test = False
        use_random = False
    elif test_method == 'random':
        clean_test = False
        use_random = True
    else:
        raise NotImplementedError
    
    
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            target = target.to(device)
            
            if clean_test:
                images = clip_normalize(images)
                image_features = model.encode_image(images)
            else:
                target_index = target.detach().cpu().numpy()
                text_of_classes = [class_names[i] for i in target_index]
                # use the first prompt template
                text_of_target_class = [prompt_templates[0].format(class_name) for class_name in text_of_classes]
                text_tokens = clip.tokenize(text_of_target_class).to(device)
                
                batch_size = images.size(0)
                noise = torch.randn(batch_size, 100).to(device)
                text_embedding = model.encode_text(text_tokens)
                sec_emb = text_embedding
                gen_image, _ = generator(noise, sec_emb)
                random_noise = torch.randn_like(gen_image)
                if use_random:
                    delta_im = random_noise
                else:
                    delta_im = gen_image
                
                norm_type = args.norm_type
                epsilon = args.epsilon
                if norm_type == "l2":
                    temp = torch.norm(delta_im.view(delta_im.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                    delta_im = delta_im * epsilon / temp
                elif norm_type == "linf":
                    delta_im = torch.clamp(delta_im, -epsilon / 255., epsilon / 255)  # torch.Size([16, 3, 256, 256])

                delta_im = delta_im.to(images.device)
                delta_im = F.interpolate(delta_im, (images.shape[-2], images.shape[-1]))
                
                # add delta(noise) to image
                images_adv = torch.clamp(images + delta_im, min=0, max=1)
                images_adv = clip_normalize(images_adv)
                
                image_features = model.encode_image(images_adv)
                
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100 

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()     
    parser.add_argument('--checkpoint', default="/remote-home/songtianwei/research/unlearn_multimodal/output/train_generator_max_loss/checkpoint_epoch_10.pth")   
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)   
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--clip_model', default='ViT-B/32', type=str)
    
    parser.add_argument('--test_method', default='clean', choices=['clean', 'adversarial', 'random'])
    # noise limit
    parser.add_argument('--norm_type', default='l2', choices=['l2', 'linf'])
    parser.add_argument('--epsilon', default=8, type=int)
    # dataset 
    parser.add_argument('--dataset', default='CIFAR10', choices=['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet'])
    # config overload
    parser.add_argument('--poisoned_ratio', default=1.0, type=float)
    
    args = parser.parse_args()
    
    main(args)