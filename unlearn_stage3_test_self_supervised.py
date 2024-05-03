# torch
import torch

# sys
import os
import argparse


# local
from models.self_supervised.simclr import (
    SimCLRStage1, SimCLRStage2, Loss
)
from utils.data_utils import (
    ContrastivePairDataset, ContrastivePairPoisonDataset,
    load_pair_dataset, load_class_dataset,
    create_simple_loader,
    contrastive_train_transform, contrastive_test_transform
)
from utils.os_utils import (
    create_folder, join_path, record_result
)
from utils.ue_util import AverageMeter
from tqdm import tqdm
import json


# stage 1: pretrain
def train_pretrain(train_dataset, args):
    DEVICE = torch.device(args.device)
    # 每次训练计算图改动较小使用，在开始前选取较优的基础算法（比如选择一种当前高效的卷积算法）
    torch.backends.cudnn.benchmark = True

    batch_size = args.pretrain_batch_size
    save_path = args.stage1_path
    max_epoch = args.pretrain_epoch
    
    train_data=torch.utils.data.DataLoader(train_dataset ,batch_size=batch_size, shuffle=True, num_workers=16 , drop_last=True)

    model = SimCLRStage1().to(DEVICE)
    lossLR= Loss().to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    for epoch in range(1,max_epoch+1):
        model.train()
        total_loss = 0
        for batch,(imgL,imgR,labels) in enumerate(train_data):
            imgL,imgR,labels=imgL.to(DEVICE),imgR.to(DEVICE),labels.to(DEVICE)

            _, pre_L=model(imgL)
            _, pre_R=model(imgR)

            loss=lossLR(pre_L,pre_R,batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch", epoch, "batch", batch, "loss:", loss.detach().item())
            total_loss += loss.detach().item()

        print("epoch loss:",total_loss/len(train_dataset)*batch_size)

        with open(os.path.join(save_path, "stage1_loss.txt"), "a") as f:
            f.write(str(total_loss/len(train_dataset)*batch_size) + " ")

        if epoch % 10==0:
            torch.save(model.state_dict(), os.path.join(save_path, 'model_stage1_epoch' + str(epoch) + '.pth'))


# stage 2: finetune
def train_finetune(args):
    torch.backends.cudnn.benchmark = True
    
    device = torch.device(args.device)  
    print("current device:", device)
    save_path = args.stage2_path
    stage1_max_epoch = args.pretrain_epoch
    stage2_max_epoch = args.finetune_epoch
    batch_size = args.finetune_batch_size

    # load dataset for train and eval
    train_dataset, test_dataset = load_class_dataset(args.dataset, train_transform=contrastive_train_transform, test_transform=contrastive_test_transform)
    train_loader, test_loader = create_simple_loader(train_dataset), create_simple_loader(test_dataset)

    pretrain_path = join_path(args.stage1_path, "model_stage1_epoch" + str(stage1_max_epoch) + ".pth")

    model = SimCLRStage2(num_class=len(train_dataset.classes)).to(device)
    model.load_state_dict(torch.load(pretrain_path, map_location='cpu'),strict=False)
    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)

    result = []
    
    if hasattr(train_dataset, 'class_to_idx'):
        class_to_idx_dict = train_dataset.class_to_idx
    else:
        class_to_idx_dict = {
            "airplane": 0, "bird": 1, "car": 2, "cat": 3, "deer": 4, "dog": 5, "horse": 6, "monkey": 7, "ship": 8, "truck": 9
        }
    idx_to_class_dict = dict(zip(class_to_idx_dict.values(), class_to_idx_dict.keys()))
    
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1,stage2_max_epoch+1):
        model.train()
        total_loss=0
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        for batch, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            pred = model(data)

            loss = loss_criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("epoch",epoch,"loss:", total_loss / len(train_dataset)* batch_size)
        with open(os.path.join(save_path, "stage2_loss.txt"), "a") as f:
            f.write(str(total_loss / len(train_dataset)* batch_size) + " ")

        if epoch % 5==0:
            torch.save(model.state_dict(), os.path.join(save_path, 'model_stage2_epoch' + str(epoch) + '.pth'))

            model.eval()
            correct, total = 0, 0
            
            
            class_correct_dict = {k:{'correct_num':0, 'total_num':0, 'correct_rate':0} for k,v in class_to_idx_dict.items()}

        
            with torch.no_grad():
                print("batch", " " * 1, "top1 acc", " " * 1, "top5 acc")
                total_loss, total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0, 0
                for batch, (data, labels) in enumerate(test_loader):
                    data, labels = data.to(device), labels.to(device)
                    logits = model(data)

                    total_num += data.size(0)
                    prediction = torch.argsort(logits, dim=-1, descending=True)
                    top1_acc = torch.sum((prediction[:, 0:1] == labels.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    top5_acc = torch.sum((prediction[:, 0:5] == labels.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    total_correct_1 += top1_acc
                    total_correct_5 += top5_acc
                    
                    _, predicted = torch.max(logits.data, 1)
                    loss_test = criterion(logits, labels)
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
            
            # record result
            record = {}
            record['epoch'] = epoch
            record['train_acc'] = acc_meter.avg
            record['train_loss'] = loss_meter.avg
            record['test_acc'] = acc
            record['test_acc_top1'] = acc
            record['test_acc_top5'] = total_correct_5 / total_num
            record['test_loss'] = loss_test.item()
            record['test_class_acc'] = class_correct_dict
            
            tqdm.write('Clean Accuracy Top1 %.2f, Top5 %.2f' % (acc*100, total_correct_5 / total_num*100))
            tqdm.write('Class Accuracy: ')
            for k,v in class_correct_dict.items():
                tqdm.write(f'{k}: {v["correct_rate"]:.2f}', end=' ')
            tqdm.write('\n')
            
            result.append(record)
            
    return result



def main(args):
    
    if args.poisoned:
        noise = torch.load(args.noise_path, map_location=args.device)
        train_dataset = ContrastivePairPoisonDataset(args.dataset, noise, contrastive_transform = contrastive_train_transform)
        args.output_dir = join_path(args.output_dir, 'poisoned')
    else:
        train_dataset = ContrastivePairDataset(args.dataset, contrastive_transform = contrastive_train_transform)
        args.output_dir = join_path(args.output_dir, 'natural')
    
    
    stage1_result_path = f'{args.output_dir}/stage1/'
    create_folder(stage1_result_path)
    stage2_result_path = f'{args.output_dir}/stage2_new/'
    create_folder(stage2_result_path)
    
    args.stage1_path = stage1_result_path
    args.stage2_path = stage2_result_path
    
    if args.stage == 'stage1' or args.stage == 'all':
        train_pretrain(train_dataset, args)
    if args.stage == 'stage2' or args.stage == 'all':
        result = train_finetune(args)
        result_save_path = stage2_result_path
        
        create_folder(result_save_path)
        record_result(result, result_save_path)
    
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()       
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'stl10', 'imagenet-cifar10'])
    parser.add_argument('--poisoned', action='store_true')
    parser.add_argument('--noise_path', default= './output/train_g_unlearn/cat_noise.pt')
    parser.add_argument('--output_dir', default='./output/unlearn_stage3_test_self_supervised')
    
    # training settings
    parser.add_argument('--distributed', action='store_true')   # 采用多卡训练
    # training stage
    parser.add_argument('--stage', default='stage2', choices=['all','stage1','stage2'])

    # training config
    parser.add_argument('--pretrain_batch_size', default=400, type=int)
    parser.add_argument('--finetune_batch_size', default=400, type=int)
    parser.add_argument('--pretrain_epoch', default=1000, type=int)
    parser.add_argument('--finetune_epoch', default=200, type=int)
    parser.add_argument('--test_train_type', default='self_supervised')
    args = parser.parse_args()

    main(args)
