import torch
import clip
import os
from torchvision.datasets import  MNIST, CIFAR10, CIFAR100, ImageNet
import numpy as np
from tqdm import tqdm

from torchvision import transforms

from dataset import create_dataset, create_sampler, create_loader, normalize_fn
from models.model_gan_generator import NetG
import utils

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load('ViT-B/32', device)
# make sure to convert the model parameters to fp32
model = model.float()
model = model.to(device) 

myTransform = transforms.Compose([
    transforms.Resize((224,224)),
    
    transforms.ToTensor(),
    # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

myNormalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


generator_checkpoint = "/remote-home/songtianwei/research/unlearn_multimodal/output/train_generator_max_loss/checkpoint_epoch_10.pth"
generator = NetG()
generator = generator.to(device)
checkpoint = torch.load(generator_checkpoint, map_location=device)
generator.load_state_dict(checkpoint['model'])
generator.to(device)



# cifar10 = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=myTransform)
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=myTransform)
# imageNet = ImageNet(root="/remote-home/songtianwei/research/unlearn_multimodal/data/imagenet", split='val', transform=myTransform)
# cifar10_loader = torch.utils.data.DataLoader(cifar10, batch_size=64, shuffle=False)
cifar100_loader = torch.utils.data.DataLoader(cifar100, batch_size=64, shuffle=False)
# imageNet_loader = torch.utils.data.DataLoader(imageNet, batch_size=64, shuffle=False)
# loader = cifar10_loader
loader = cifar100_loader
# loader = imageNet_loader


# from https://github.com/openai/CLIP/blob/main/data/prompts.md
mnist_classes = ['0','1','2','3','4','5','6','7','8','9',]
mnist_templates = ['a photo of the number: "{}".',]
cifar10_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck',]
cifar100_classes = cifar100.classes
# imagenet_classes = imageNet.classes
cifar10_templates = [
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


def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


class_map = {'MNIST': mnist_classes, 'CIFAR10': cifar10_classes}
template_map = {'MNIST': mnist_templates, 'CIFAR10': cifar10_templates}

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

cifar10_classes = class_map['CIFAR10']
cifar10_templates = template_map['CIFAR10']


zeroshot_weights = zeroshot_classifier(cifar100_classes, cifar10_templates)

import torch.nn.functional as F

use_adversarial = True
use_random = False

with torch.no_grad():
    top1, top5, n = 0., 0., 0.
    for i, (images, target) in enumerate(tqdm(loader)):
        images = images.to(device)
        target = target.to(device)
        
        # for noise generate
        target_index = target.detach().cpu().numpy()
        text_of_classes = [cifar100_classes[i] for i in target_index]
        text_of_target_class = [cifar10_templates[0].format(class_name) for class_name in text_of_classes]
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
        
        norm_type = 'linf'
        epsilon = 8
        if norm_type == "l2":
            temp = torch.norm(delta_im.view(delta_im.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            delta_im = delta_im * epsilon / temp
        elif norm_type == "linf":
            delta_im = torch.clamp(delta_im, -epsilon / 255., epsilon / 255)  # torch.Size([16, 3, 256, 256])

        delta_im = delta_im.to(images.device)
        delta_im = F.interpolate(delta_im, (images.shape[-2], images.shape[-1]))
        
        # add delta(noise) to image
        images_adv = torch.clamp(images + delta_im, min=0, max=1)
        images_adv = myNormalize(images_adv)
        
        # predict
        if use_adversarial:
            image_features = model.encode_image(images_adv)
        else:
            images = myNormalize(images)
            image_features = model.encode_image(images)
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
     
    
