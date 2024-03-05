import os
import pickle

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import  MNIST, CIFAR10, CIFAR100, ImageNet, STL10, GTSRB

import numpy as np
import json
from PIL import Image

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

def load_class_dataset(dataset_name, train_transform=None, test_transform=None):
    # zero-shot test dataset
    if dataset_name == 'MNIST':
        train_dataset = MNIST(root=os.path.expanduser("~/.cache"), download=True, train=True, transform=train_transform)
        test_dataset = MNIST(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=test_transform)
    elif dataset_name == 'CIFAR10' or dataset_name == 'cifar10':
        train_dataset = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=True, transform=train_transform)
        test_dataset = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=test_transform)
    elif dataset_name == 'CIFAR100':
        train_dataset = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True, transform=train_transform)
        test_dataset = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=test_transform)
    elif dataset_name == 'ImageNet':
        train_dataset = ImageNet(root="/remote-home/songtianwei/research/unlearn_multimodal/data/imagenet", split='train', transform=train_transform)
        test_dataset = ImageNet(root="/remote-home/songtianwei/research/unlearn_multimodal/data/imagenet", split='val', transform=test_transform)
    elif dataset_name == 'STL10' or dataset_name == 'stl10':
        train_dataset = STL10(root=os.path.expanduser("~/.cache"), download=True, split='train', transform=train_transform)
        test_dataset = STL10(root=os.path.expanduser("~/.cache"), download=True, split='test', transform=test_transform)
    elif dataset_name == 'GTSRB':
        train_dataset = GTSRB(root=os.path.expanduser("~/.cache"), download=True, split='train', transform=train_transform)
        test_dataset = GTSRB(root=os.path.expanduser("~/.cache"), download=True, split='test', transform=test_transform)
        prompt_template = "A photo of a traffic sign of {}."
    else:
        raise NotImplementedError
    
    return train_dataset, test_dataset

def load_poison_dataset(dataset_name, noise, train_transform=None, test_transform=None):
    """load cifar-10 noise to its training dataset, specific class

    Args:
        dataset_name (str): name (cifar10)
        noise_path (Tensor): noise([5000,3,32,32])
        train_transform (Transforms): train_transform
        test_transform (Transforms): test_transform
    """
    if dataset_name == 'cifar10' or dataset_name == 'CIFAR10':
        unlearnable_train_dataset = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=True, transform=train_transform)
        test_dataset = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=test_transform)
    elif dataset_name == 'stl10' or dataset_name == 'STL10':
        unlearnable_train_dataset = STL10(root=os.path.expanduser("~/.cache"), download=True, split='train', transform=train_transform)
        test_dataset = STL10(root=os.path.expanduser("~/.cache"), download=True, split='test', transform=test_transform)
        
    target_poison_class_name = "cat"
    
    if dataset_name == 'cifar10' or dataset_name == 'CIFAR10':
        target_label = unlearnable_train_dataset.class_to_idx[target_poison_class_name]
        train_lables = unlearnable_train_dataset.targets
    elif dataset_name == 'stl10' or dataset_name == 'STL10':
        class_to_idx_dict = {
            "airplane": 0, "bird": 1, "car": 2, "cat": 3, "deer": 4, "dog": 5, "horse": 6, "monkey": 7, "ship": 8, "truck": 9
        }
        target_label = class_to_idx_dict[target_poison_class_name]
        train_lables = unlearnable_train_dataset.labels

    image_shape = unlearnable_train_dataset.data[0].shape
    
    # print("The shape of noise is: ", noise.shape)
    # print("The shape of image is: ", image_shape)
    if noise.shape[1:] != image_shape:
        raise ValueError("The shape of noise is not equal to the shape of image.")
        
    # perturb_noise = noise.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
    perturb_noise = noise.mul(255).clamp_(0, 255).to('cpu').numpy()
    unlearnable_train_dataset.data = unlearnable_train_dataset.data.astype(np.float32)
    noise_idx = 0
    for i in range(len(unlearnable_train_dataset)):
        label = train_lables[i]
        if label == target_label:  # poison the specific class 'cat'
            # print(perturb_noise[noise_idx].shape)
            # print(unlearnable_train_dataset.data[i].shape)
            unlearnable_train_dataset.data[i] += perturb_noise[noise_idx]
            unlearnable_train_dataset.data[i] = np.clip(unlearnable_train_dataset.data[i], a_min=0, a_max=255)
            noise_idx += 1
    unlearnable_train_dataset.data = unlearnable_train_dataset.data.astype(np.uint8)

    if noise_idx != noise.shape[0]:
        raise ValueError("The number of noise is not equal to the number of target class.")

    return unlearnable_train_dataset, test_dataset

class jsonDataset(Dataset):
    def __init__(self,json_path,text_transform=None, img_transform=None, contain_index=False):
        self.json_data = json.load(open(json_path,'r'))
        self.text_transform = text_transform
        self.img_transform = img_transform
        self.contain_index = contain_index
    def __getitem__(self,idx):
        sample = self.json_data[idx]
        text, img_path, index = sample['caption'], sample['image_path'], sample['index']
        img = Image.open(img_path)
        img = img.convert('RGB')
        if self.text_transform:
            text = self.text_transform(text)
        if self.img_transform:
            img = self.img_transform(img)
        if self.contain_index:
            return img, text, index
        else:
            return img, text
    def __len__(self):
        return len(self.json_data)

class jsonPoisonDataset(Dataset):
    def __init__(self,json_path,noise_path,text_transform=None, img_transform=None, contain_index=False):
        self.json_data = json.load(open(json_path,'r'))
        self.noise_list = torch.load(noise_path)

        self.text_transform = text_transform
        self.img_transform = img_transform
        
        self.contain_index = contain_index
    def __getitem__(self,idx):
        sample = self.json_data[idx]
        text, img_path, index, label_class = sample['caption'], sample['image_path'], sample['index'], sample['class']
        img = Image.open(img_path)
        img = img.convert('RGB')
        if self.text_transform:
            text = self.text_transform(text)
        if self.img_transform:
            img = self.img_transform(img)
        
        if label_class == 'cat':
            the_noise_index = index % len(self.noise_list)
            the_noise = self.noise_list[the_noise_index]
            the_noise = the_noise.to(img.device)
            the_img = torch.clamp(img + the_noise, min=0, max=1)
        else:
            the_img = img
            
        if self.contain_index:
            return the_img, text, index
        else:
            return the_img, text
    def __len__(self):
        return len(self.json_data)

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

import torch
from torchvision import datasets, transforms

class ImageTextDatasetFromSupervisedDataset(Dataset):
    def __init__(self, dataset_name, split='train', transform=None) -> None:
        super().__init__()
        self.supervised_train_dataset, self.supervised_test_dataset = load_class_dataset(dataset_name, None)
        
        # Transformations for the dataset
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
            
        if split == 'train':    
            self.dataset = self.supervised_train_dataset
        else:
            self.dataset = self.supervised_test_dataset
        
        self.image_list, self.text_list = self.construct_dataset(self.dataset)
    
    def construct_prompt(self, label):
        text_prompt = "This is a picture of a {}"
        text = text_prompt.format(label)
        return text 
    
    def construct_dataset(self, dataset):
        image_list, text_list = [], []
        for index in range(len(dataset)):
            image, label = self.dataset[index]
            label_name = self.dataset.classes[label]
            text = self.construct_prompt(label_name)
            image_list.append(image)
            text_list.append(text)
        return image_list, text_list
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image, text = self.image_list[index], self.text_list[index]
        if self.transform:
            image = self.transform(image)
        return image, text

class ImageTextDatasetFromSupervisedDatasetPoison(Dataset):
    def __init__(self, dataset_name, split='train', transform=None, noise_path=None) -> None:
        super().__init__()
        noise = torch.load(noise_path)
        self.supervised_train_dataset, self.supervised_test_dataset = load_poison_dataset(dataset_name, noise, None)
        
        # Transformations for the dataset
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
            
        if split == 'train':    
            self.dataset = self.supervised_train_dataset
        else:
            self.dataset = self.supervised_test_dataset
        
        self.image_list, self.text_list = self.construct_dataset(self.dataset)

    def construct_prompt(self, label):
        text_prompt = "This is a picture of a {}"
        text = text_prompt.format(label)
        return text 
    
    def construct_dataset(self, dataset):
        image_list, text_list = [], []
        for index in range(len(dataset)):
            image, label = self.dataset[index]
            label_name = self.dataset.classes[label]
            text = self.construct_prompt(label_name)
            image_list.append(image)
            text_list.append(text)
        return image_list, text_list
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image, text = self.image_list[index], self.text_list[index]
        
        if self.transform:
            image = self.transform(image)
        return image, text

# Dataset used for contrastive learning
class ContrastivePairDataset(Dataset):
    
    def __init__(self, dataset_name,contrastive_transform=None ,train_transform=None, test_transform=None, target_transform=None) -> None:
        super().__init__()
        self.supervised_train_dataset, self.supervised_test_dataset = load_class_dataset(dataset_name, train_transform, test_transform)
        self.contrastive_transform = contrastive_transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.supervised_train_dataset)
    
    def __getitem__(self, index):
        img, target= self.supervised_train_dataset.data[index], self.supervised_train_dataset.targets[index]
        img = Image.fromarray(img)

        if self.contrastive_transform is not None:
            imgL = self.contrastive_transform(img)
            imgR = self.contrastive_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return imgL, imgR, target
# Dataset poisoning for contrastive learning
class ContrastivePairPoisonDataset(Dataset):
    
    def __init__(self, dataset_name, noise, contrastive_transform=None ,train_transform=None, test_transform=None, target_transform=None) -> None:
        super().__init__()
        self.supervised_train_dataset, self.supervised_test_dataset = load_poison_dataset(dataset_name, noise, train_transform, test_transform)
        self.contrastive_transform = contrastive_transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.supervised_train_dataset)
    
    def __getitem__(self, index):
        img, target= self.supervised_train_dataset.data[index], self.supervised_train_dataset.targets[index]
        img = Image.fromarray(img)

        if self.contrastive_transform is not None:
            imgL = self.contrastive_transform(img)
            imgR = self.contrastive_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return imgL, imgR, target

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


def create_my_loader(trainDataset, testDataset):
    clean_train_loader = DataLoader(dataset=trainDataset, batch_size=128,
                                shuffle=False, pin_memory=True,
                                drop_last=False, num_workers=12)
    clean_test_loader = DataLoader(dataset=testDataset, batch_size=128,
                                    shuffle=False, pin_memory=True,
                                    drop_last=False, num_workers=12)
    return clean_train_loader, clean_test_loader

def create_simple_loader(dataset):
    loader = DataLoader(dataset, batch_size=256,
                                shuffle=True, pin_memory=True,
                                drop_last=False, num_workers=12)
    return loader

from torchvision import transforms


# transforms

ToTensorTrans = transforms.Compose([
    transforms.ToTensor()
])

To244TensorTrans = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# transform for simclr
# train transform for simclr
contrastive_train_transform  = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])
# test transform for simclr
contrastive_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])