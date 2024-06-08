from models.ResNet import ResNet18

from utils.data_utils import (
    load_poison_dataset, load_class_dataset,
    To32TensorTrans, To244TensorTrans,
    transform_supervised_train,transform_supervised_test
)
from utils.noise_utils import (
    limit_noise
)
from utils.record_utils import (
    record_result_supervised, AverageMeter,
    RecordSupervised,
    jsonRecord
)
from utils.os_utils import (
    create_folder
)

from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.models import (
    resnet18 as torchvision_resnet18,
    resnet50 as torchvision_resnet50,
    resnet101 as torchvision_resnet101,
    ResNet18_Weights, ResNet50_Weights
)
from torchvision import transforms
from torch.utils.data import (DataLoader)
import torch

import argparse

import os
from pathlib import Path

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def test_supervised(trainDataset, testDataset, args):
    train_loader = DataLoader(dataset=trainDataset, batch_size=512,
                                    shuffle=True, pin_memory=True,
                                    drop_last=False, num_workers=12)
    test_loader = DataLoader(dataset=testDataset, batch_size=512,
                                    shuffle=False, pin_memory=True,
                                    drop_last=False, num_workers=12)
    device = args.device
    # model = ResNet18()
    if args.pretrained:
        print("Using pretrained model - ResNet18 - IMAGENET1K_V1")
        model = torchvision_resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        #     print("Using pretrained model - ResNet18 - IMAGENET1K_V1")
        #     model = torchvision_resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # elif args.dataset == 'stl10':
        #     print("Using pretrained model - ResNet50 - IMAGENET1K_V1")
        #     model = torchvision_resnet101(weights=ResNet50_Weights.IMAGENET1K_V1)
    else:
        print("Using scratched model - ResNet18")
        model = torchvision_resnet18(pretrained=False)
        # if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        #     print("Using scratched model - ResNet18")
        #     model = torchvision_resnet18(pretrained=False)
        # elif args.dataset == 'stl10':
        #     print("Using scratched model - ResNet50")
        #     model = torchvision_resnet101(pretrained=False)
        
    if args.dataset == 'cifar10':
        model.fc = torch.nn.Linear(512, 10)
    elif args.dataset == 'stl10':
        model.fc = torch.nn.Linear(512, 10)
    elif args.dataset == 'cifar100':
        model.fc = torch.nn.Linear(512, 100)
    
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    lr = args.lr
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=0.0005, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0)
    
    if hasattr(trainDataset, 'class_to_idx'):
        # For cifar10 and cifar100, the class_to_idx is already defined
        class_to_idx_dict = trainDataset.class_to_idx
    else:
        # For stl10, we need to define the class_to_idx
        class_to_idx_dict = {
            "airplane": 0, "bird": 1, "car": 2, "cat": 3, "deer": 4, "dog": 5, "horse": 6, "monkey": 7, "ship": 8, "truck": 9
        }
        
    idx_to_class_dict = dict(zip(class_to_idx_dict.values(), class_to_idx_dict.keys()))
    myRecord = RecordSupervised()
    
    for epoch in range(args.max_epoch):
        # Training Stage
        model.train()
        
        acc_meter_train = AverageMeter()
        loss_meter_train = AverageMeter()
        
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
            
            acc_meter_train.update(acc)
            loss_meter_train.update(loss.item())
            
            if args.poisoned:
                pbar.set_description("Epoch %d, Acc %.2f, Loss: %.2f, Poisoned" % (epoch, acc_meter_train.avg*100, loss_meter_train.avg))
            else:
                pbar.set_description("Epoch %d, Acc %.2f, Loss: %.2f, Clean" % (epoch, acc_meter_train.avg*100, loss_meter_train.avg))
        
        scheduler.step()
        # Eval Stage
        model.eval()
        acc_meter_test = AverageMeter()
        loss_meter_test = AverageMeter()
        
        class_correct_dict = {k:{'correct_num':0, 'total_num':0, 'correct_rate':0} for k,v in class_to_idx_dict.items()}

        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                loss = criterion(logits, labels)
                
                _, predicted = torch.max(logits.data, 1)
                acc = (predicted == labels).sum().item()/labels.size(0)
                
                acc_meter_test.update(acc)
                loss_meter_test.update(loss.item())
                
                # 统计class_correct_dict正确的类别个数
                for j in range(labels.size(0)):
                    label = labels[j]
                    class_label = idx_to_class_dict[label.item()]
                    class_correct_dict[class_label]['correct_num'] += (predicted[j] == label).item()
                    class_correct_dict[class_label]['total_num'] += 1
            
        # record the result
        for k,v in class_correct_dict.items():
            if v['total_num'] != 0:
                class_correct_dict[k]['correct_rate'] = v['correct_num']/v['total_num']
        
        # record result
        myRecord.add_one_record(epoch, acc_meter_train, loss_meter_train, acc_meter_test ,loss_meter_test, class_correct_dict)
        
        tqdm.write('Clean Accuracy %.2f' % (acc_meter_test.avg*100))
        tqdm.write('Class Accuracy: ')
        for k,v in class_correct_dict.items():
            tqdm.write(f'{k}: {v["correct_num"]} / {v["total_num"]} = {v["correct_rate"]:.2f}', end='\t')
        tqdm.write('\n')
    
    return myRecord.get_result()





def main(args):
    # Create save folder
    # the experiment save path: ./${output_dir}/${dataset}/${natural/poisoned}/${pretrained/scratch}${poisoned_source}
    args.output_dir = os.path.join(args.output_dir, args.dataset)

    if args.poisoned:
        args.output_dir = os.path.join(args.output_dir, "poisoned")
        if args.pretrained:
            args.output_dir = os.path.join(args.output_dir, "pretrained")
        else:
            args.output_dir = os.path.join(args.output_dir, "scratched")
        noise_path = args.noise_path
        noise_clip_version = noise_path.split('/')[-4]  # get noise training source
        noise_type_version = noise_path.split('/')[-2]  # get noise type version (sample/class)
        args.noise_clip_version = noise_clip_version
        args.noise_type_version = noise_type_version
        args.output_dir = os.path.join(args.output_dir, f"noise_of_{args.finetune_dataset}_{noise_clip_version}")
        args.output_dir = os.path.join(args.output_dir, noise_type_version)
    else:
        args.output_dir = os.path.join(args.output_dir, "natural")
        if args.pretrained:
            args.output_dir = os.path.join(args.output_dir, "pretrained")
        else:
            args.output_dir = os.path.join(args.output_dir, "scratched")
    create_folder(args.output_dir)
    
    train_transform = To32TensorTrans
    test_transform = To32TensorTrans
        
    if args.poisoned:
        noise = torch.load(args.noise_path, map_location=args.device)
        
        # test fix noise 
        if args.fix_noise:
            tgt_shape = noise[0].shape
            noise = torch.randn_like(noise[0])
            noise = torch.stack([noise] * 5000)
            noise = limit_noise(noise, noise_shape=tgt_shape, norm_type="l2", epsilon=8, device=args.device)
            
        poison_train_dataset, test_dataset = load_poison_dataset(args.dataset, noise, target_poison_class_name=args.poison_class_name, train_transform=train_transform, test_transform=test_transform)
        train_dataset = poison_train_dataset
    else:
        # poison_train_dataset, test_dataset = load_poison_dataset(args.dataset, noise, target_poison_class_name=args.poison_class_name, train_transform=train_transform, test_transform=test_transform)
        natural_train_dataset, test_dataset = load_class_dataset(args.dataset, train_transform=train_transform, test_transform=test_transform)
        train_dataset = natural_train_dataset
    
    result = test_supervised(train_dataset, test_dataset, args)
    myJsonRecord = jsonRecord(os.path.join(args.output_dir, "result.json"))
    myJsonRecord.save_args(args)
    myJsonRecord.save_exp_res(result)
    
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()       
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'stl10', 'cifar100'])
    parser.add_argument('--poisoned', action='store_true')
    parser.add_argument('--noise_path', default= './output/unlearn_stage2_generate_noise/ViT-B-16/cifar10/noise_gen1_ViT-B-16_cifar10_all.pt')
    parser.add_argument('--output_dir', default='./output/unlearn_stage3_test_supervised/')
    
    parser.add_argument('--poison_class_name', default='all', choices=['all', 'airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'])
    # For train  
    parser.add_argument('--max_epoch', default=40, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--transform', default='default', choices=['default', 'supervised'])
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--backbone', default='resnet18', choices=['resnet18', 'resnet50', 'resnet101'])
    
    # for model(pretrained or not)
    parser.add_argument('--pretrained', action='store_true')
    
    # fix noise
    parser.add_argument('--fix_noise', action='store_true')
    
    # used for test
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_folder', default='./output/unlearn_test_supervised/')
    
    parser.add_argument('--test_train_type', default='supervised')
    
    parser.add_argument('--finetune_dataset', default='mylaion', choices=['laion', 'cifar10', 'stl10'])
    args = parser.parse_args()

    main(args)
