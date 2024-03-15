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
    create_folder, join_path
)

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
    # 每次训练计算图改动较小使用，在开始前选取较优的基础算法（比如选择一种当前高效的卷积算法）
    torch.backends.cudnn.benchmark = True
    
    device = torch.device(args.device)  
    print("current device:", device)
    save_path = args.stage2_path
    max_epoch = args.pretrain_epoch
    batch_size = args.finetune_batch_size

    # load dataset for train and eval
    train_dataset, test_dataset = load_class_dataset(args.dataset, train_transform=contrastive_train_transform, test_transform=contrastive_test_transform)
    train_loader, test_loader = create_simple_loader(train_dataset), create_simple_loader(test_dataset)

    pretrain_path = join_path(args.stage1_path, "model_stage1_epoch" + str(max_epoch) + ".pth")

    model = SimCLRStage2(num_class=len(train_dataset.classes)).to(device)
    model.load_state_dict(torch.load(pretrain_path, map_location='cpu'),strict=False)
    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)

    for epoch in range(1,max_epoch+1):
        model.train()
        total_loss=0
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
            with torch.no_grad():
                print("batch", " " * 1, "top1 acc", " " * 1, "top5 acc")
                total_loss, total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0, 0
                for batch, (data, target) in enumerate(test_loader):
                    data, target = data.to(device), target.to(device)
                    pred = model(data)

                    total_num += data.size(0)
                    prediction = torch.argsort(pred, dim=-1, descending=True)
                    top1_acc = torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    top5_acc = torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    total_correct_1 += top1_acc
                    total_correct_5 += top5_acc

                    print("  {:02}  ".format(batch + 1), " {:02.3f}%  ".format(top1_acc / data.size(0) * 100),
                          "{:02.3f}%  ".format(top5_acc / data.size(0) * 100))

                print("all eval dataset:", "top1 acc: {:02.3f}%".format(total_correct_1 / total_num * 100),
                          "top5 acc:{:02.3f}%".format(total_correct_5 / total_num * 100))
                with open(os.path.join(save_path, "stage2_top1_acc.txt"), "a") as f:
                    f.write(str(total_correct_1 / total_num * 100) + " ")
                with open(os.path.join(save_path, "stage2_top5_acc.txt"), "a") as f:
                    f.write(str(total_correct_5 / total_num * 100) + " ")



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
    stage2_result_path = f'{args.output_dir}/stage2/'
    create_folder(stage2_result_path)
    
    args.stage1_path = stage1_result_path
    args.stage2_path = stage2_result_path
    
    # train_pretrain(train_dataset, args)
    train_finetune(args)
    
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()       
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'stl10', 'imagenet-cifar10'])
    parser.add_argument('--poisoned', action='store_true')
    parser.add_argument('--noise_path', default= '/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn/cat_noise.pt')
    parser.add_argument('--output_dir', default='/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_test_self_supervised')
    
    # training config
    parser.add_argument('--pretrain_batch_size', default=400, type=int)
    parser.add_argument('--finetune_batch_size', default=400, type=int)
    parser.add_argument('--pretrain_epoch', default=1000, type=int)
    parser.add_argument('--finetune_epoch', default=200, type=int)
    args = parser.parse_args()

    main(args)
