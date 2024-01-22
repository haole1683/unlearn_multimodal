import torch

from utils.ue_util import AverageMeter
from models.ResNet import ResNet18
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def test_supervised(trainDataset, testDataset):
    train_loader = DataLoader(dataset=trainDataset, batch_size=512,
                                    shuffle=False, pin_memory=True,
                                    drop_last=False, num_workers=12)
    test_loader = DataLoader(dataset=testDataset, batch_size=512,
                                    shuffle=False, pin_memory=True,
                                    drop_last=False, num_workers=12)
    # 正常训练
    model = ResNet18()
    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, weight_decay=0.0005, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0)
    
    ship_label = trainDataset.class_to_idx['ship']
    cat_label = trainDataset.class_to_idx['cat']
    
    for epoch in range(30):
        # Train
        model.train()
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        pbar = tqdm(train_loader, total=len(train_loader))
        for images, labels in pbar:
            images, labels = images.cuda(), labels.cuda()
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
            pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))
        scheduler.step()
        # Eval
        model.eval()
        correct, total = 0, 0
        cat_correct, cat_total = 0,0
        cat_label = 3
        ship_label = 8
        ship_correct, ship_total = 0,0
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                logits = model(images)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                cat_total += (labels == cat_label).sum().item()
                cat_correct += ((predicted == labels) & (labels == cat_label)).sum().item()

                ship_total += (labels == ship_label).sum().item()
                ship_correct += ((predicted == labels) & (labels == ship_label)).sum().item()
        acc = correct / total
        cat_acc = cat_correct / cat_total
        ship_acc = ship_correct / ship_total
        tqdm.write('Clean Accuracy %.2f | Cat acc %.2f ,  Cat correct %d, Cat Total %d | Ship acc %.2f ,  Ship correct %d, Ship Total %d \n'
                % (acc*100, cat_acc *100, cat_correct, cat_total , ship_acc *100, ship_correct, ship_total))