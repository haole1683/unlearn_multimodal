from train_downstream_solo import classify
from utils.clip_util import get_clip_model

clip = get_clip_model('ViT-B/16','cuda:0')

import torch

from utils.data_utils import load_poison_dataset, load_class_dataset

noise = torch.load("/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn/cat_noise.pt")

from torchvision import transforms

myTrans = transforms.Compose([
    transforms.ToTensor()
    ])

ue_train_cifar10, cifar_test = load_poison_dataset("cifar10", noise, myTrans)

clean_train_cifar10, cifar_test = load_class_dataset("CIFAR10", myTrans)

classify(clip, ue_train_cifar10, cifar_test)