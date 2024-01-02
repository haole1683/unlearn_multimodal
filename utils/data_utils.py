import os
import pickle

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import  MNIST, CIFAR10, CIFAR100, ImageNet, STL10, GTSRB
from torchvision.datasets import ImageFolder

class CustomDataSet(Dataset):
    def __init__(
            self,
            images,
            texts,
            labs,
            ids
    ):
        self.images = images
        self.texts = texts
        self.labs = labs
        self.ids = ids

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        lab = self.labs[index]
        id = self.ids[index]
        return img, text, lab, id

    def __len__(self):
        count = len(self.texts)
        return count

def load_dataloader(name, bsz):
    train_loc = 'data/' + name + '/train.pkl'
    test_loc = 'data/' + name + '/test.pkl'
    with open(train_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        train_labels = data['label']
        train_texts = data['text']
        train_images = data['image']
        train_ids = data['ids']
    with open(test_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        test_labels = data['label']
        test_texts = data['text']
        test_images = data['image']
        test_ids = data['ids']
    imgs = {'train': train_images, 'test': test_images}
    texts = {'train': train_texts, 'test': test_texts}
    labs = {'train': train_labels, 'test': test_labels}
    ids = {'train': train_ids, 'test': test_ids}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labs=labs[x], ids=ids[x])
               for x in ['train', 'test']}

    shuffle = {'train': True, 'test': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=bsz,
                                shuffle=shuffle[x], num_workers=0) for x in ['train', 'test']}

    return dataloader


def load_dataset(name, bsz):
    train_loc = 'data/' + name + '/train.pkl'
    test_loc = 'data/' + name + '/test.pkl'
    with open(train_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        train_labels = data['label']
        train_texts = data['text']
        train_images = data['image']
        train_ids = data['ids']
    with open(test_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        test_labels = data['label']
        test_texts = data['text']
        test_images = data['image']
        test_ids = data['ids']
    imgs = {'train': train_images, 'test': test_images}
    texts = {'train': train_texts, 'test': test_texts}
    labs = {'train': train_labels, 'test': test_labels}
    ids = {'train': train_ids, 'test': test_ids}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labs=labs[x], ids=ids[x])
               for x in ['train', 'test']}

    return dataset

def load_pair_dataset(name, bsz):
    return load_dataset(name, bsz)

def load_class_dataset(dataset_name, transform):
    # zero-shot test dataset
    if dataset_name == 'MNIST':
        train_dataset = MNIST(root=os.path.expanduser("~/.cache"), download=True, train=True, transform=transform)
        test_dataset = MNIST(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=transform)
    elif dataset_name == 'CIFAR10':
        train_dataset = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=True, transform=transform)
        test_dataset = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=transform)
    elif dataset_name == 'CIFAR100':
        train_dataset = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True, transform=transform)
        test_dataset = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=transform)
    elif dataset_name == 'ImageNet':
        train_dataset = ImageNet(root="/remote-home/songtianwei/research/unlearn_multimodal/data/imagenet", split='train', transform=transform)
        test_dataset = ImageNet(root="/remote-home/songtianwei/research/unlearn_multimodal/data/imagenet", split='val', transform=transform)
    elif dataset_name == 'STL10':
        train_dataset = STL10(root=os.path.expanduser("~/.cache"), download=True, split='train', transform=transform)
        test_dataset = STL10(root=os.path.expanduser("~/.cache"), download=True, split='test', transform=transform)
    elif dataset_name == 'GTSRB':
        train_dataset = GTSRB(root=os.path.expanduser("~/.cache"), download=True, split='train', transform=transform)
        test_dataset = GTSRB(root=os.path.expanduser("~/.cache"), download=True, split='test', transform=transform)
        prompt_template = "A photo of a traffic sign of {}."
    else:
        raise NotImplementedError
    
    return train_dataset, test_dataset

def load_json_data(dataset_name):
    import json
    if dataset_name == "MNIST":
        with open("data/MNIST/mnist.json", "r") as f:
            data = json.load(f)
    elif dataset_name == "CIFAR10":
        with open("data/CIFAR10/cifar10.json", "r") as f:
            data = json.load(f)
    elif dataset_name == "CIFAR100":
        with open("data/CIFAR100/cifar100.json", "r") as f:
            data = json.load(f)
    elif dataset_name == "ImageNet":
        with open("data/ImageNet/imagenet.json", "r") as f:
            data = json.load(f)
    elif dataset_name == "STL10":
        with open("data/STL10/stl10.json", "r") as f:
            data = json.load(f)
    elif dataset_name == "GTSRB":
        with open("data/GTSRB/gtsrb.json", "r") as f:
            data = json.load(f)
    else:
        raise NotImplementedError
    return data

def load_folder_data(folder_path):
    from torchvision.datasets import ImageFolder
    import torchvision.transforms as transforms
    
    return ImageFolder(root=folder_path, transform=transforms.ToTensor())

def get_dataset_class(dataset_name):
    """Get the classes name of the dataset.

    Args:
        dataset_name (str): the name of the dataset
    """
    train_dataset = load_class_dataset(dataset_name, None)[0]
    return train_dataset.classes
    
    


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    