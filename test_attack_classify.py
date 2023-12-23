import argparse
import torch
import clip
import os
from torchvision.datasets import  MNIST, CIFAR10, CIFAR100, ImageNet, STL10, GTSRB
import numpy as np
from tqdm import tqdm
import ruamel.yaml as yaml
import clip
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from models.model_gan_generator import NetG
from utils.patch_utils import de_normalize, clamp_patch, mask_generation, patch_initialization
from utils.noise_utils import gen_perturbation
from utils.record import record_result
from utils.clip_util import _convert_image_to_rgb, clip_transform, clip_normalize, prompt_templates, zeroshot_classifier
from utils.load_data import load_class_dataset
from utils.evaluate import test_linear_probe

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def test_linear_probe_noise(trainloader,testloader,device,model,delta_im, arg):
    def get_features(dataloader,delta_im, attack=False):
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                if attack:
                    delta_im = delta_im.to(device)
                    images = images.to(device)
                    images = torch.clamp(images + delta_im, min=0, max=1)
                    images = clip_normalize(images)
                features = model.encode_image(images.to(device))

                all_features.append(features)
                all_labels.append(labels)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

    # Calculate the image features
    train_features, train_labels = get_features(trainloader,delta_im, attack=False)
    test_features, test_labels = get_features(testloader, delta_im,attack=True)

    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")
    return accuracy

def test_linear_probe_patch(trainloader,testloader,device,model,uap_noise, mask, arg):
    def get_features(dataloader, attack=False):
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                if attack:
                    new_shape = images.shape
                    f_x = torch.mul(mask.type(torch.FloatTensor),
                                    uap_noise.type(torch.FloatTensor)) + torch.mul(
                        1 - mask.expand(new_shape).type(torch.FloatTensor), images.type(torch.FloatTensor))

                    f_x = f_x.to(device)  
                    f_x = clip_normalize(f_x)      
                    images = f_x
                features = model.encode_image(images.to(device))

                all_features.append(features)
                all_labels.append(labels)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

    # Calculate the image features
    train_features, train_labels = get_features(trainloader, attack=False)
    test_features, test_labels = get_features(testloader, attack=True)

    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")
    return accuracy


def main(args):
    device = args.device
    
    # load clip model
    model, preprocess = clip.load(args.clip_model, device)
    # make sure to convert the model parameters to fp32
    model = model.float()
    model = model.to(device) 
    
    # load generator
    generator = NetG()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint['model'])
    generator = generator.to(device)
    
    # load dataset
    train_dataset, test_dataset = load_class_dataset(args.dataset, clip_transform)
    
    batch_size = args.batch_size
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    class_names = test_dataset.classes
    zeroshot_weights = zeroshot_classifier(model, device, class_names, prompt_templates)
    
    if args.attack_type == "universal":
        prompt = "An image"
        prompt_tokens = clip.tokenize([prompt]).to(device)
        prompt_embedding = model.encode_text(prompt_tokens)
        delta_im = gen_perturbation(generator, prompt_embedding, torch.zeros((1, 3, 224, 224)).to(device), args)
        torch.save(delta_im, "/remote-home/songtianwei/research/unlearn_multimodal/delta_im.pt")
     
    ################## linear probe #########################
    print("linear probe")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    if args.baseline == "my":
        linear_probe_result = test_linear_probe_noise(train_loader, test_dataloader, device, model, delta_im, args)
    elif args.baseline == "advclip":
        uap_noise_path = "/remote-home/songtianwei/research/advCLIP/AdvCLIP/output/uap/gan_patch/ViT-B16/nus-wide/0.03/uap_gan_96.34_17.pt"
        uap_noise = torch.load(uap_noise_path, map_location=device) # [3,224,224]
        uap_noise = clamp_patch(args, uap_noise)
        uap_noise.to(device)
        
        patch = patch_initialization(args)
        mask, applied_patch, x, y = mask_generation(args, patch)
        applied_patch = torch.from_numpy(applied_patch)
        mask = torch.from_numpy(mask)

        # add the uap
        linear_probe_result = test_linear_probe_patch(train_loader, test_dataloader, device, model,uap_noise, mask, args)
    print(f"Linear probe result: {linear_probe_result:.2f}")
    
    ################## zero shot ###########################
    print("zero shot")
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(test_dataloader)):
            images = images.to(device)
            target = target.to(device)
            
            if args.baseline == "clean":
                images = clip_normalize(images)
                image_features = model.encode_image(images)
                
            elif args.baseline == "advclip":
                uap_noise_path = "/remote-home/songtianwei/research/advCLIP/AdvCLIP/output/uap/gan_patch/ViT-B16/nus-wide/0.03/uap_gan_96.34_17.pt"
                uap_noise = torch.load(uap_noise_path, map_location=device) # [3,224,224]
                uap_noise = clamp_patch(args, uap_noise)
                uap_noise.to(device)
                
                patch = patch_initialization(args)
                mask, applied_patch, x, y = mask_generation(args, patch)
                applied_patch = torch.from_numpy(applied_patch)
                mask = torch.from_numpy(mask)

                # add the uap
                new_shape = images.shape
                f_x = torch.mul(mask.type(torch.FloatTensor),
                                uap_noise.type(torch.FloatTensor)) + torch.mul(
                    1 - mask.expand(new_shape).type(torch.FloatTensor), images.type(torch.FloatTensor))

                f_x = f_x.to(device)  
                f_x = clip_normalize(f_x)      
                image_features = model.encode_image(f_x)
            elif args.baseline == "my":
                if args.attack_type == "universal":
                    images_adv = torch.clamp(images + delta_im, min=0, max=1)
                    images_adv = clip_normalize(images_adv)
                    image_features = model.encode_image(images_adv)
                elif args.attack_type == "sample":
                    target_index = target.detach().cpu().numpy()
                    text_of_classes = [class_names[i] for i in target_index]
                    # use the first prompt template
                    rand_index = np.random.randint(0, len(prompt_templates))
                    fix_index = True
                    if fix_index:   # fix the index of rand index to 0
                        rand_index = 0
                    prompt_template = None
                    if prompt_template:
                        text_of_target_class = [prompt_template.format(class_name) for class_name in text_of_classes]
                    else:
                        text_of_target_class = [prompt_templates[rand_index].format(class_name) for class_name in text_of_classes]
                    text_tokens = clip.tokenize(text_of_target_class).to(device)
                    text_embedding = model.encode_text(text_tokens)
                    delta_im = gen_perturbation(generator, text_embedding, images, args)
                    # add delta(noise) to image
                    images_adv = torch.clamp(images + delta_im, min=0, max=1)
                    images_adv = clip_normalize(images_adv)
                    image_features = model.encode_image(images_adv)
                    
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100 

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")
    
    result = {
        "top1": top1,
        "top5": top5
    }
    
    
    
    return result

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--checkpoint', default="/remote-home/songtianwei/research/unlearn_multimodal/output/cur_universal_gen_flickr_ViT-B-16/checkpoint_epoch_75.pth")   
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)   
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--clip_model', default='ViT-B/16', type=str, choices=['RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'ViT-B/16'])
    
    # use universarial attack
    parser.add_argument("--attack_type", default="universal", choices=["universal", "sample"])
    
    parser.add_argument('--baseline', default='my', choices=['my', 'advclip', 'clean'])
    parser.add_argument('--norm_type', default='l2', choices=['l2', 'linf'])
    parser.add_argument('--epsilon', default=8, type=int)
    # dataset 
    parser.add_argument('--dataset', default='CIFAR10', choices=['MNIST', 'STL10', 'CIFAR10',
                                                                  'CIFAR100','GTSRB','ImageNet',
                                                                  'NUS-WIDE', 'Pascal', 'Wikipedis', 'XmediaNet'
                                                                  ])
    parser.add_argument('--attack_ratio', default=1.0, type=float)
    
    args = parser.parse_args()
    
    result = main(args)
    
    record_result(args, result)