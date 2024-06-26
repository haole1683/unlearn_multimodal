import argparse
import torch
import clip
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from torchvision.utils import save_image

from models.model_gan_generator import NetG
from utils.patch_utils import de_normalize, clamp_patch, mask_generation, patch_initialization
from utils.noise_utils import gen_perturbation
from utils.record import record_result
from utils.clip_util import _convert_image_to_rgb, clip_transform, clip_normalize, prompt_templates, zeroshot_classifier
from utils.data_utils import load_class_dataset
from utils.evaluate import test_linear_probe, test_linear_probe_noise, test_linear_probe_patch, accuracy, zero_shot, test_linear_probe_unlearn, zero_shot_with_each_class_acc

def test_zero_shot(model, clip_version=None):
    if isinstance(model, str):
        device = "cuda:0"
        model, preprocess = clip.load(model, device)
    else:
        device = model.text_projection.device
    model.eval()
    # load dataset
    dataset_name = "CIFAR10"
    train_dataset, test_dataset = load_class_dataset(dataset_name, clip_transform)
    batch_size = 64
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    class_names = test_dataset.classes
    zeroshot_weights = zeroshot_classifier(model, device, class_names, prompt_templates)

    process_fn = clip_normalize
    
    ################## zero shot ###########################
    # print("Start zero shot")
    # top1, top5 = zero_shot(test_dataloader, model, zeroshot_weights, device, process_fn=process_fn)
    # print(f"Zero shot result: top1: {top1}, top5: {top5}")
    
    top1, top5, class_acc = zero_shot_with_each_class_acc(test_dataloader, test_dataset, model, zeroshot_weights, device, process_fn=process_fn)
    print(f"Zero shot result: top1: {top1}, top5: {top5}")
    
    ################## linear probe #########################
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # print("Start linear probe")
    # linear_probe_result = test_linear_probe(train_loader, test_dataloader, device, model, args, process_fn=process_fn)
    # print(f"Linear probe result: {linear_probe_result:.2f}")
    
    result = {
        # "linear-probe": 
        #     linear_probe_result,
        "zero-shot":{
            "top1": top1,
            "top5": top5
        },
        "class_acc": class_acc
    }
    return result
    
def main(args):
    device = args.device
    
    # load clip model
    model, preprocess = clip.load(args.clip_model, device)
    # make sure to convert the model parameters to fp32
    my_clip_pretrain_path = "/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_finetune_clip/model_ViT-B_16_poison_epoch9.pth"
    checkpoint = torch.load(my_clip_pretrain_path, map_location=device)
    print(f"Load clip model from {my_clip_pretrain_path}")
    if "model" in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model = model.float()
    model = model.to(device) 
    
    # load generator
    generator = NetG()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    print(f"Load generator from {args.checkpoint}")
    generator.load_state_dict(checkpoint['model'])
    generator = generator.to(device)
    
    # load dataset
    train_dataset, test_dataset = load_class_dataset(args.dataset, clip_transform)
    
    batch_size = args.batch_size
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    class_names = test_dataset.classes
    zeroshot_weights = zeroshot_classifier(model, device, class_names, prompt_templates)

    if args.baseline == "clean":
        process_fn = clip_normalize
        
    elif args.baseline == "advclip":
        patch = patch_initialization(args)
        mask, applied_patch, x, y = mask_generation(args, patch)
        applied_patch = torch.from_numpy(applied_patch)
        mask = torch.from_numpy(mask)
        uap_noise_path = "//remote-home/songtianwei/research/advCLIP/AdvCLIP/output/uap/gan_patch/ViT-B16/nus-wide/0.03/uap_gan_95.73_10.pt"
        uap_noise = torch.load(uap_noise_path, map_location=device) # [3,224,224]
        uap_noise = clamp_patch(args, uap_noise)
        uap_noise = de_normalize(uap_noise)
        uap_noise.to(device)
        
        def process_fn(images):
            new_shape = images.shape
            f_x = torch.mul(mask.type(torch.FloatTensor),
                            uap_noise.type(torch.FloatTensor)) + torch.mul(
                1 - mask.expand(new_shape).type(torch.FloatTensor), images.type(torch.FloatTensor))

            f_x = f_x.to(device)  
            f_x = clip_normalize(f_x)      
            return f_x

    elif args.baseline == "my":
        if args.attack_type == "universal":
            prompt = "An image of an bird"
            prompt_tokens = clip.tokenize([prompt]).to(device)
            prompt_embedding = model.encode_text(prompt_tokens)
            delta_im = gen_perturbation(generator, prompt_embedding, torch.zeros((1, 3, 224, 224)).to(device), args)
            delta_im = delta_im.to(device)
            torch.save(delta_im, "/remote-home/songtianwei/research/unlearn_multimodal/delta_im.pt")
            def process_fn(images):
                images_adv = torch.clamp(images + delta_im, min=0, max=1)
                images_adv = clip_normalize(images_adv)
                return images_adv
        elif args.attack_type == "sample":
            def process_fn(images, target):
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
                return image_features
    
    else:
        raise NotImplementedError
    
    ################## zero shot ###########################
    print("Start zero shot")
    top1, top5 = zero_shot(test_dataloader, model, zeroshot_weights, device, process_fn=process_fn)
    print(f"Zero shot result: top1: {top1}, top5: {top5}")
    
    ################## linear probe #########################
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    print("Start linear probe")
    linear_probe_result = test_linear_probe(train_loader, test_dataloader, device, model, args, process_fn=process_fn)
    print(f"Linear probe result: {linear_probe_result:.2f}")
    
    ################# linear probe unlearn (attach attack in training set) #####################
    print("Start linear probe unlearn")
    linear_probe_result_unlearn = test_linear_probe_unlearn(train_loader, test_dataloader, device, model, args, process_fn=process_fn)
    print(f"Linear probe result unlearn: {linear_probe_result_unlearn:.2f}")
    
    ################## record result ########################
    result = {
        "linear-probe": 
            linear_probe_result,
        "linear-probe-unlearn":
            linear_probe_result_unlearn,
        "zero-shot":{
            "top1": top1,
            "top5": top5
        }
    }
    
    return result

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--checkpoint', default="/remote-home/songtianwei/research/unlearn_multimodal/output/text_targeted_gen_nus-wide_CIFAR100_ViT-B-16/checkpoint_epoch_15.pth")   
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)   
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--clip_model', default='ViT-B/16', type=str, choices=['RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'ViT-B/16'])
    
    # use universarial attack
    parser.add_argument("--attack_type", default="universal", choices=["universal", "sample"])
    
    parser.add_argument('--baseline', default='clean', choices=['my', 'advclip', 'clean'])
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