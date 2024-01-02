import os
import random
import csv
import clip
import torch
import datetime
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
from utils.data_utils import load_dataset, load_dataloader
from utils.model import model, NonLinearClassifier
from utils.patch_utils import patch_initialization, mask_generation
from utils.evaluate import cross_evaluate, pat_cross_evaluate, solo_test, solo_adv_test, solo_fr_test, solo_adv_test_noise,solo_fr_noise_test

def load_config():
    parser = argparse.ArgumentParser(description='Test AdvCLIP')
    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--dataset', type=str, default='nus-wide', choices=['nus-wide', 'pascal', 'wikipedia', 'xmedianet', 'imagenet', 'gtsrb', 'cifar10', 'stl10'])
    parser.add_argument('--sup_dataset', type=str, default='nus-wide', choices=['nus-wide', 'pascal', 'wikipedia', 'xmedianet'])
    parser.add_argument('--mode', type=str, default='gan_patch', choices=['gan_patch', 'opt_patch'])
    parser.add_argument('--down_type', type=str, default='solo', choices=['cross', 'solo'])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--device', default="cuda:0", type=str, help='which gpu the code runs on')
    parser.add_argument('--victim', default='ViT-B/16', choices=['ViT-L/14', 'ViT-B/16', 'ViT-B/32', 'RN50', 'RN101'])
    parser.add_argument('--noise_percentage', type=float, default=0.03)
    args = parser.parse_args()
    return args

def classify_test(encoder, F, test_loader, delta_im, device):

    F.to(device)
    encoder.to(device)
    encoder.eval()
    clean_acc_t1, clean_acc_t5 = solo_test(encoder, F, test_loader, device)
    adv_acc_t1, adv_acc_t5 = solo_adv_test_noise(encoder, F, test_loader,delta_im, device)
    print("########################  Normal Performance ########################")
    print('Top1 test acc: %.4f, Top5 test acc: %.4f' % (clean_acc_t1, clean_acc_t5))
    print("######################## Attack Performance ########################")
    print('Top1 solo_adv_test acc: %.4f, Top5 solo_adv_test acc: %.4f' % (adv_acc_t1, adv_acc_t5))
    fooling_rate = solo_fr_noise_test(encoder, F, test_loader, delta_im, device)
    decline_t1 = ((clean_acc_t1 - adv_acc_t1) / clean_acc_t1) * 100
    decline_t5 = ((clean_acc_t5 - adv_acc_t5) / clean_acc_t5) * 100

    print('Top1 test acc: %.4f, Top1 solo_adv_test acc: %.4f, Fooling rate: %.4f'
        % (clean_acc_t1, adv_acc_t1, fooling_rate))

    return clean_acc_t1, adv_acc_t1, decline_t1, fooling_rate, clean_acc_t5, adv_acc_t5, decline_t5


def solo_final_test(args, num_class, feature_dim, clip_model, test_loader,delta_im, device):

    model_ft = NonLinearClassifier(feat_dim=feature_dim, num_classes=num_class)
    model_ft_path = "/remote-home/songtianwei/research/advCLIP/AdvCLIP/output/solo_model/ViT-B/16/nus-wide/ViT-B_16_nus-wide_74.752_20.pth"
    weights = torch.load(model_ft_path, map_location=device)

    weights_dict = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    model_ft.load_state_dict(weights_dict)
    model_ft.to(device)

    # load uap


    clean_acc_t1, adv_acc_t1, decline_t1, fooling_rate, clean_acc_t5, adv_acc_t5, decline_t5 = classify_test(clip_model, model_ft,
                                                                                                             test_loader,
                                                                                                             delta_im,
                                                                                                             device)
    print('Clean downstream accuracy: %.4f%%' % (clean_acc_t1))
    print('Adv downstream accuracy: %.4f%%' % (adv_acc_t1))
    print('Decline accuracy rate: %.4f%%' % (decline_t1))
    print('Downstream fooling rate: %.4f%%' % (fooling_rate))

    return clean_acc_t1, adv_acc_t1, decline_t1, fooling_rate, clean_acc_t5, adv_acc_t5, decline_t5

def solo_write(args, now_time, clean_acc_t1, adv_acc_t1, decline_t1, fooling_rate, clean_acc_t5, adv_acc_t5, decline_t5):

    final_log_save_path = os.path.join('output', 'results', str(args.down_type), str(args.mode))
    if not os.path.exists(final_log_save_path):
        os.makedirs(final_log_save_path)

    final_result = []

    final_result_ = {"now_time": now_time,
                     "attack_type": args.down_type,
                     "victim": str(args.victim),
                     "sup_dataset": str(args.sup_dataset),
                     "down_dataset": str(args.dataset),
                     "noise_percentage": str(round(args.noise_percentage, 4)),
                     "clean_acc_t1": round(clean_acc_t1, 4),
                     "clean_acc_t5": round(clean_acc_t5, 4),
                     "decline_t1": round(decline_t1, 4),
                     "adv_acc_t1": round(adv_acc_t1, 4),
                     "adv_acc_t5": round(adv_acc_t5, 4),
                     "decline_t5": round(decline_t5, 4),
                     "fooling_rate": round(fooling_rate, 4)}
    final_result.append(final_result_)
    header = ["now_time", "attack_type", "victim", "sup_dataset", "down_dataset",
              "noise_percentage", "clean_acc_t1", "clean_acc_t5", "decline_t1",
              "adv_acc_t1", "adv_acc_t5", "decline_t5", "fooling_rate"]

    with open(final_log_save_path + '/all_final_results.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(final_result)
    print("write success!")

def cross_final_test(args, clip_model, test_loader, num_class, feature_dim, device):

    model_ft = model(num_class=num_class, img_dim=feature_dim, text_dim=feature_dim, mid_dim=256, feature_dim=feature_dim).to(
        device)

    model_ft_root = os.path.join('output', 'model', str(args.victim), str(args.dataset))
    model_ft_path = [Path(model_ft_root) / ckpt for ckpt in os.listdir(Path(model_ft_root)) if ckpt.endswith(".pt")][0]
    weights = torch.load(model_ft_path)

    weights_dict = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    model_ft.load_state_dict(weights_dict)
    model_ft.to(device)

    print("########################  Normal Performance ########################")
    # test the clean performance of downstream tasks
    img2text, text2img, t_imgs, t_txts, t_labels = cross_evaluate(clip_model, model_ft, test_loader, device)
    map = (img2text + text2img) / 2
    print(f'img2text: {img2text*100:.4f}, text2img: {text2img*100:.4f}')


    print("######################## Attack Performance ########################")

    # load uap
    uap_root = os.path.join('output', 'uap', str(args.mode), str(args.victim), str(args.sup_dataset),
                            str(args.noise_percentage))
    uap_path = [Path(uap_root) / ckpt for ckpt in os.listdir(Path(uap_root)) if ckpt.endswith("20.pt")][0]

    uap = torch.load(uap_path)

    patch = patch_initialization(args)
    mask, applied_patch, x, y = mask_generation(args, patch)
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)

    adv_img2text, adv_text2img, adv_t_imgs, adv_t_txts, adv_t_labels, fr = pat_cross_evaluate(clip_model, model_ft, test_loader, uap, mask,
                                                                      device)

    adv_map = (adv_img2text + adv_text2img) / 2

    print(f'Adv_Normal mAP: {adv_map*100:.4f}, img2text: {adv_img2text*100:.4f}, text2img: {adv_text2img*100:.4f}, fooling rate: {fr:.4f}')
    print( f'ASR_i: {(img2text - adv_img2text) * 100:.4f}, ASR_t: {(text2img - adv_text2img) * 100:.4f}')

    return img2text, text2img, map, adv_img2text, adv_text2img, adv_map, fr

def cross_write(args, now_time, i_map,t_map, map, fr, p_i_map, p_t_map, p_map):

    final_log_save_path = os.path.join('output', 'results', str(args.down_type), str(args.mode))
    if not os.path.exists(final_log_save_path):
        os.makedirs(final_log_save_path)

    final_result = []
    final_result_ = {"now_time": now_time,
                     "victim": str(args.victim),
                     "sup_dataset": str(args.sup_dataset),
                     "dataset": str(args.dataset),
                     "noise_percentage": str(round(args.noise_percentage, 4)),
                     "i_map": round(i_map * 100, 4),
                     "t_map": round(t_map * 100, 4),
                     "map": round(map * 100, 4),
                     "fooling_rate": round(fr, 4),
                     "p_i_map": round(p_i_map * 100, 4),
                     "p_t_map": round(p_t_map * 100, 4),
                     "p_map": round(p_map * 100, 4),
                     "d_i_t": round(((i_map - p_i_map) / i_map) * 100, 4),
                     "d_t_i": round(((t_map - p_t_map) / t_map) * 100, 4),
                     "d_map": round(((map - p_map) / map) * 100, 4),
                     }
    final_result.append(final_result_)
    header = ["now_time", "victim", "sup_dataset", "dataset", "noise_percentage", "i_map", "t_map",  "map", "fooling_rate",
              "p_i_map", "p_t_map", "p_map","d_i_t", "d_t_i", "d_map"]

    with open(final_log_save_path + '/all_final_results.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(final_result)
    print("write success!")

import torch.nn.functional as F
def gen_perturbation(generator,text_embedding, images, args):
    batch_size = images.shape[0]
    sec_emb = text_embedding
    device = args.device
    noise = torch.randn(batch_size, 100).to(device)
    gen_image, _ = generator(noise, sec_emb)
    delta_im = gen_image
    norm_type = args.norm_type
    epsilon = args.epsilon
    if norm_type == "l2":
        temp = torch.norm(delta_im.view(delta_im.shape[0], -1), dim=1).view(-1, 1, 1, 1)
        delta_im = delta_im * epsilon / temp
    elif norm_type == "linf":
        delta_im = torch.clamp(delta_im, -epsilon / 255., epsilon / 255)  # torch.Size([16, 3, 256, 256])

    delta_im = delta_im.to(images.device)
    delta_im = F.interpolate(delta_im, (images.shape[-2], images.shape[-1]))
    
    return delta_im


if __name__ == '__main__':

    args = load_config()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # set device for CLIP
    USE_CUDA = torch.cuda.is_available()
    device = torch.device(args.device if USE_CUDA else "cpu")

    dataloaders = load_dataloader(args.dataset, args.batch_size)
    train_loader = dataloaders['train']
    test_loader = dataloaders['test']
    print(len(train_loader), len(test_loader))

    if args.dataset == 'wikipedia':
        num_class = 10
    elif args.dataset == 'pascal':
        if args.down_type == 'cross':
            num_class = 20
        elif args.down_type == 'solo':
            num_class = 21
    elif args.dataset == 'xmedianet':
        num_class = 100
    elif args.dataset == 'nus-wide':
        num_class = 81

    if args.victim == 'ViT-L/14':
        feature_dim = 768
    elif args.victim == 'RN50':
        feature_dim = 1024
    elif args.victim == 'ViT-B/16' or args.victim == 'ViT-B/32' or args.victim == 'RN101':
        feature_dim = 512
    else:
        feature_dim = 512


    clip_model, preprocess = clip.load(args.victim, device=device)  # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']

    now_time = datetime.datetime.now().strftime('%Y_%m_%d')
    
    delta_im = torch.load("/remote-home/songtianwei/research/unlearn_multimodal/delta_im.pt", map_location=device)


    if args.down_type == 'cross':

        img2text, text2img, map, adv_img2text, adv_text2img, adv_map, fr = cross_final_test(args, clip_model, test_loader, num_class, feature_dim, device)
        if args.save == True:
            cross_write(args, now_time, img2text, text2img, map, fr, adv_img2text, adv_text2img, adv_map)

    elif args.down_type == 'solo':

        clean_acc_t1, adv_acc_t1, decline_t1, fooling_rate, clean_acc_t5, adv_acc_t5, decline_t5 = solo_final_test(args, num_class, feature_dim, clip_model, test_loader,delta_im, device)
        if args.save == True:
            solo_write(args, now_time, clean_acc_t1, adv_acc_t1, decline_t1, fooling_rate, clean_acc_t5, adv_acc_t5,
                       decline_t5)
