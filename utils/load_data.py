import os
import pickle
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import  MNIST, CIFAR10, CIFAR100, ImageNet, STL10, GTSRB


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