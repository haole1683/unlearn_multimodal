import torch

from utils.ue_util import AverageMeter
from models.ResNet import ResNet18
from torchvision.models import resnet18 as torchvision_resnet18
from torchvision import transforms
from torch.utils.data import (DataLoader)
import torch
from utils.data_utils import (
    load_poison_dataset, load_class_dataset
)
from tqdm import tqdm
import torchvision.transforms as transforms
import argparse

import json
import os

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

train_transform = transforms.Compose([
    transforms.Resize(size=(96, 96)),
    transforms.RandomCrop(size=(64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 0.5)),
    transforms.ColorJitter(brightness=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize(size=(96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
])

def test_supervised(trainDataset, testDataset, args):
    train_loader = DataLoader(dataset=trainDataset, batch_size=512,
                                    shuffle=True, pin_memory=True,
                                    drop_last=False, num_workers=12)
    test_loader = DataLoader(dataset=testDataset, batch_size=512,
                                    shuffle=False, pin_memory=True,
                                    drop_last=False, num_workers=12)
    device = args.device
    # 正常训练
    # model = ResNet18()
    model = torchvision_resnet18(pretrained=True)
    
    model.fc = torch.nn.Linear(in_features=512, out_features=10)
    # if args.dataset == 'cifar10':
    #     model.linear = torch.nn.Linear(512, 10)
    # elif args.dataset == 'stl10':
    #     model.linear = torch.nn.Linear(4608, 10)
    # else:
    #     model.linear = torch.nn.Linear(512, 10)
    
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, weight_decay=0.0005, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0)

    result = []
    
    if hasattr(trainDataset, 'class_to_idx'):
        class_to_idx_dict = trainDataset.class_to_idx
    else:
        class_to_idx_dict = {
            "airplane": 0, "bird": 1, "car": 2, "cat": 3, "deer": 4, "dog": 5, "horse": 6, "monkey": 7, "ship": 8, "truck": 9
        }
        
    idx_to_class_dict = dict(zip(class_to_idx_dict.values(), class_to_idx_dict.keys()))
    
    for epoch in range(40):
        # Train
        model.train()
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        pbar = tqdm(train_loader, total=len(train_loader))
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == labels).sum().item()/labels.size(0)
            acc_meter.update(acc)
            loss_meter.update(loss.item())
            if args.poisoned:
                pbar.set_description("Epoch %d, Acc %.2f, Loss: %.2f, Poisoned" % (epoch, acc_meter.avg*100, loss_meter.avg))
            else:
                pbar.set_description("Epoch %d, Acc %.2f, Loss: %.2f, Clean" % (epoch, acc_meter.avg*100, loss_meter.avg))
        scheduler.step()
        # Eval
        model.eval()
        correct, total = 0, 0
        correct_dict = {}
        
        class_correct_dict = {k:{'correct_num':0, 'total_num':0, 'correct_rate':0} for k,v in class_to_idx_dict.items()}

        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 统计class_correct_dict正确的类别个数
                for j in range(labels.size(0)):
                    label = labels[j]
                    class_label = idx_to_class_dict[label.item()]
                    class_correct_dict[class_label]['correct_num'] += (predicted[j] == label).item()
                    class_correct_dict[class_label]['total_num'] += 1
                    
        for k,v in class_correct_dict.items():
            if v['total_num'] != 0:
                class_correct_dict[k]['correct_rate'] = v['correct_num']/v['total_num']
                    
        acc = correct / total
        
        record = {}
        record['epoch'] = epoch
        record['acc'] = acc
        record['class_acc'] = class_correct_dict
        
        tqdm.write('Clean Accuracy %.2f' % (acc*100))
        tqdm.write('Class Accuracy: ')
        for k,v in class_correct_dict.items():
            tqdm.write(f'{k}: {v["correct_rate"]:.2f}', end=' ')
        tqdm.write('\n')
        
        result.append(record)
    return result

def record_result(result, folder_path):
    import os
    file_path = os.path.join(folder_path, "result.txt")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(file_path, 'w') as f:
        for record in result:
            f.write(f'Epoch: {record["epoch"]}\n')
            f.write(f'Accuracy: {record["acc"]}\n')
            f.write(f'Class Accuracy: \n')
            for k,v in record['class_acc'].items():
                f.write(f'{k}: {v["correct_num"]},{v["total_num"]}, {v["correct_rate"]:.2f} | ')
            f.write('\n')
    
    # save as json
    file_path = os.path.join(folder_path, "result.json")
    with open(file_path, 'w') as f:
        json.dump(result, f)
    

def main(args):
    
    myTrans = transforms.Compose([
        # transforms.Resize((32,32)),
        transforms.ToTensor()
    ])
    
    
    if args.poisoned:
        noise = torch.load(args.noise_path, map_location=args.device)
        poison_train_dataset, test_dataset = load_poison_dataset(args.dataset, noise, test_transform)
        train_dataset = poison_train_dataset
    else:
        natural_train_dataset, test_dataset = load_class_dataset(args.dataset, train_transform)
        train_dataset = natural_train_dataset
    
    
    result = test_supervised(train_dataset, test_dataset, args)
    
    args.output_dir = os.path.join(args.output_dir, args.dataset)
    
    if args.poisoned:
        poison_result_path = f'{args.output_dir}/poison/'
        result_save_path = poison_result_path
    else:
        natural_result_path = f'{args.output_dir}/natural/'
        result_save_path = natural_result_path
    
    
    record_result(result_save_path, natural_result_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()       
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'stl10', 'imagenet-cifar10'])
    parser.add_argument('--poisoned', action='store_true')
    parser.add_argument('--noise_path', default= '/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn/cat_noise_RN50.pt')
    parser.add_argument('--output_dir', default='/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_test_supervised')
    args = parser.parse_args()

    main(args)
