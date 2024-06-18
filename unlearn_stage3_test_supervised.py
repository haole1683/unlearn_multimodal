from models.ResNet import ResNet18

from utils.data_utils import (
    load_poison_dataset, load_class_dataset,
    To32TensorTrans, To244TensorTrans,
    transform_supervised_train_64, transform_supervised_test_96,
    transform_supervised_train_224, transform_supervised_test_224
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
    ResNet18_Weights, ResNet50_Weights, ResNet101_Weights,
    VisionTransformer,vit_b_16,
    ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights, ViT_L_32_Weights
)
from torchvision.models import (
    vit_b_16, vit_b_32, vit_l_16, vit_l_32
)
from torch.utils.data import (DataLoader)
import torch

import argparse

import os
from pathlib import Path

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def test_supervised(trainDataset, testDataset, args):
    if args.backbone == 'ViT-B_16':
        args.batch_size = args.batch_size // 4  # 512 CUDA out of memory
    print(f"Batch Size: {args.batch_size}")
    train_loader = DataLoader(dataset=trainDataset, batch_size=args.batch_size,
                                    shuffle=True, pin_memory=True,
                                    drop_last=False, num_workers=12)
    test_loader = DataLoader(dataset=testDataset, batch_size=args.batch_size,
                                    shuffle=False, pin_memory=True,
                                    drop_last=False, num_workers=12)
    device = args.device

    if args.pretrained: # use pretrained model
        if args.backbone == 'resnet18':
            print("Using pretrained model - ResNet18 - IMAGENET1K_V1")
            model = torchvision_resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif args.backbone == 'resnet50':
            print("Using pretrained model - ResNet50 - IMAGENET1K_V1")
            model = torchvision_resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif args.backbone == 'resnet101':
            print("Using pretrained model - ResNet101 - IMAGENET1K_V1")
            model = torchvision_resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        elif args.backbone == 'ViT-B_16':
            print("Using pretrained model - ViT-B_16 - IMAGENET2")
            model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        elif args.backbone == 'ViT-B_32':
            print("Using pretrained model - ViT-B_32 - IMAGENET2")
            model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
    else:   # use scratch model
        if args.backbone == 'resnet18':
            print("Using scratch model - ResNet18")
            model = torchvision_resnet18()
        elif args.backbone == 'resnet50':
            print("Using scratch model - ResNet50")
            model = torchvision_resnet50()
        elif args.backbone == 'resnet101':
            print("Using scratch model - ResNet101")
            model = torchvision_resnet101()
        elif args.backbone == 'ViT-B_16':
            print("Using scratch model - ViT-B_16")
            model = vit_b_16()
        elif args.backbone == 'ViT-B_32':
            print("Using scratch model - ViT-B_32")
            model = vit_b_32()
    
    if args.backbone.startswith('resnet'):
        fc_in_dim = model.fc.in_features
        if args.dataset == 'cifar10':
            model.fc = torch.nn.Linear(fc_in_dim, 10)
        elif args.dataset == 'stl10':
            model.fc = torch.nn.Linear(fc_in_dim, 10)
        elif args.dataset == 'cifar100':
            model.fc = torch.nn.Linear(fc_in_dim, 100)
    
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
        
        if args.save_model and epoch % 50 == 0:
            model_save_path = os.path.join(args.output_dir, f"model_epoch{epoch}.pth")
            torch.save(model.state_dict(), model_save_path)
        
        tqdm.write('Clean Accuracy %.2f' % (acc_meter_test.avg*100))
        tqdm.write('Class Accuracy: ')
        for k,v in class_correct_dict.items():
            tqdm.write(f'{k}: {v["correct_num"]} / {v["total_num"]} = {v["correct_rate"]:.2f}', end='\t')
        tqdm.write('\n')
    
    return myRecord.get_result()





def main(args):
    # Create save folder
    # the experiment save path: ./${output_dir}/${dataset}/${model}/${natural/poisoned}/${pretrained/scratch}${poisoned_source}
    args.output_dir = os.path.join(args.output_dir, args.dataset, args.backbone)

    if args.poisoned:
        args.output_dir = os.path.join(args.output_dir, "poisoned")
        if args.pretrained:
            args.output_dir = os.path.join(args.output_dir, "pretrained")
        else:
            args.output_dir = os.path.join(args.output_dir, "scratched")
        noise_path = args.noise_path
        # eg: ./output/unlearn_stage2_generate_noise/ViT-B_16/stl10/classWise/noise_gen1_ViT-B-16_stl10_all.pt
        noise_clip_version = noise_path.split('/')[-4]  # get noise training source clip model
        noise_type_version = noise_path.split('/')[-2]  # get noise type version (sample/class)
        noise_of_dataset = noise_path.split('/')[-3]  # get noise of dataset
        if noise_of_dataset != args.dataset:
            print(f"Warning: The noise_of_dataset({noise_of_dataset}) is not equal to dataset ({args.dataset})")
            exit(0)
        args.noise_clip_version = noise_clip_version
        args.noise_type_version = noise_type_version
        args.output_dir = os.path.join(args.output_dir, f"noise_of_{noise_clip_version}")
        args.output_dir = os.path.join(args.output_dir, noise_type_version)
    else:
        args.output_dir = os.path.join(args.output_dir, "natural")
        print(args.pretrained)
        if args.pretrained:
            print("!11")
            args.output_dir = os.path.join(args.output_dir, "pretrained")
        else:
            print("!22")
            args.output_dir = os.path.join(args.output_dir, "scratched")
    create_folder(args.output_dir)
    print('create folder in ', args.output_dir)
    
    if args.backbone.startswith('resnet'):
        train_transform =  transform_supervised_train_64
        test_transform = transform_supervised_test_96
    elif args.backbone.startswith('ViT'):
        train_transform = transform_supervised_train_224
        test_transform = transform_supervised_test_224
        
    if args.poisoned:
        noise = torch.load(args.noise_path, map_location=args.device)
        poison_train_dataset, test_dataset = load_poison_dataset(args.dataset, noise, target_poison_class_name=args.poison_class_name, train_transform=train_transform, test_transform=test_transform)
        train_dataset = poison_train_dataset
    else:
        natural_train_dataset, test_dataset = load_class_dataset(args.dataset, train_transform=train_transform, test_transform=test_transform)
        train_dataset = natural_train_dataset
    
    result = test_supervised(train_dataset, test_dataset, args)
    myJsonRecord = jsonRecord(os.path.join(args.output_dir, "result.json"))
    myJsonRecord.save_args(args)
    myJsonRecord.save_exp_res(result)
    
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()       
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dataset', default='stl10', choices=['cifar10', 'stl10', 'cifar100'])
    parser.add_argument('--poisoned', action='store_true')
    parser.add_argument('--noise_path', default= './output/unlearn_stage2_generate_noise_temp1/ViT-B-32(500E)/stl10/classWise/noise_gen1_ViT-B-32_stl10_all.pt')
    parser.add_argument('--output_dir', default='./output/unlearn_stage3_test_supervised/')
    parser.add_argument('--poison_class_name', default='all', choices=['all', 'airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'])
    
    # For train  
    parser.add_argument('--max_epoch', default=101, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--transform', default='default', choices=['default', 'supervised'])
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--backbone', default='ViT-B_16', choices=['resnet18', 'resnet50', 'resnet101', 'ViT-B_16', 'ViT-B_32'])
    
    # for model(pretrained or not)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--test_train_type', default='supervised')
    parser.add_argument('--finetune_dataset', default='coco', choices=['laion', 'cifar10', 'stl10', 'coco'])
    
    parser.add_argument('--save_model', default=False, type=bool)
    args = parser.parse_args()

    main(args)
