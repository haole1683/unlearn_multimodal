import os
import torch
import time
from tqdm import tqdm
from utils.model import NonLinearClassifier

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # correct = pred.eq(target.expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def test(encoder, classifier, test_loader, device):
    top1_accuracy = 0
    top5_accuracy = 0
    encoder.eval()
    classifier.eval()

    cat_correct, cat_total = 0,0
    cat_label = 3
    ship_label = 8
    ship_correct, ship_total = 0,0

    with torch.no_grad():
        for counter, (x_batch, y_batch) in enumerate(tqdm(test_loader)):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            h = encoder.encode_image(x_batch.squeeze())
            x_in = h.view(h.size(0), -1)
            # x_in = torch.tensor(x_in, dtype=torch.float)
            logits = classifier(x_in.float())
            top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

            labels = y_batch
            predicted = logits.argmax(1)
            cat_total += (labels == cat_label).sum().item()
            cat_correct += ((predicted == labels) & (labels == cat_label)).sum().item()

            ship_total += (labels == ship_label).sum().item()
            ship_correct += ((predicted == labels) & (labels == ship_label)).sum().item()
            
        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)

        cat_acc = cat_correct / cat_total
        ship_acc = ship_correct / ship_total
        tqdm.write('Clean Top1 Accuracy %.2f Top5 Acc %.2f | Cat acc %.2f ,  Cat correct %d, Cat Total %d | Ship acc %.2f ,  Ship correct %d, Ship Total %d \n'
                % (top1_accuracy.item(), top5_accuracy.item(),cat_acc *100, cat_correct, cat_total , ship_acc *100, ship_correct, ship_total))

    return top1_accuracy.item(), top5_accuracy.item()


def classify(encoder, train_loader, test_loader):
    device = "cuda:0"
    epochs = 20
    
    num_class = 10 # For cifar
    feat_dim = encoder.visual.output_dim
    
    
    
    F = NonLinearClassifier(feat_dim=feat_dim, num_classes=num_class)
    F.to(device)
    encoder.to(device)
    # classifier
    my_optimizer = torch.optim.Adam(F.parameters(), lr=0.005, weight_decay=0.0008)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optimizer, gamma=0.96)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    F.train()
    encoder.eval()

    for epoch in range(epochs):
        start = time.time()
        top1_train_accuracy = 0
        pbar = tqdm(train_loader, total=len(train_loader))
        # clean_acc_t1, clean_acc_t5 = test(encoder, F, test_loader, device)
        for counter, (x_batch, y_batch) in enumerate(pbar):
            my_optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            h = encoder.encode_image(x_batch.squeeze())
            downstream_input = h.view(h.size(0), -1)
            logits = F(downstream_input.float())
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]
            loss.backward()
            my_optimizer.step()
            
            pbar.set_description("Acc %.2f Loss: %.2f" % (top1_train_accuracy.item()*100, loss.item()))

        end = time.time()
        F.train()
        clean_acc_t1, clean_acc_t5 = test(encoder, F, test_loader, device)
        my_lr_scheduler.step()
        top1_train_accuracy /= (counter + 1)
        print('Epoch [%d/%d], Top1 train acc: %.4f, Top1 test acc: %.4f, Time: %.4f'
              % (epoch + 1, epochs, top1_train_accuracy.item(), clean_acc_t1, (end - start)))

    return clean_acc_t1, clean_acc_t5

