import csv
import os
import time
import torch

def write_csv(result_dict, args):
    csv_path = "/remote-home/songtianwei/research/unlearn_multimodal/record/record.csv"
    header = ["now_time", "victim", "sup_dataset", "dataset", "noise_percentage", "i_map", "t_map",  "map", "fooling_rate",
              "p_i_map", "p_t_map", "p_map","d_i_t", "d_t_i", "d_map"]
    if not os.path.exists(csv_path):
        # if not exist, create the file and write the header
        with open(csv_path, 'w') as f:
            pass
        
def record_result(args, result):
    csv_path = "./record/record.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write("gen_dataset, gen_clip_model, gen_epoch, gen_clip_loss, checkpoint_path, dataset, model, attack_type, norm_type, epsilon, top1, top5\n")
    zero_shot_result = result["zero-shot"]
    top1 = zero_shot_result["top1"]
    top5 = zero_shot_result["top5"]
    
    if args.attack_type != "clean":
        checkpoint_path = args.checkpoint
        # gen_flickr_ViT-B-16
        folder_name = checkpoint_path.split("/")[-2]
        gen_dataset = folder_name.split("_")[1]
        gen_clip_model = folder_name.split("_")[2]
        gen_epoch = checkpoint_path.split("/")[-1].split(".")[0].split("_")[-1]
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        gen_clip_loss = checkpoint["loss"] if checkpoint.get("loss") else "None"
    else:
        gen_dataset = "None"
        gen_clip_model = "None"
        gen_epoch = "None"
        checkpoint_path = "None"
        gen_clip_loss = "None"
    dataset = args.dataset
    model = args.clip_model
    attack_type = args.attack_type
    norm_type = args.norm_type
    epsilon = args.epsilon
    with open(csv_path, "a") as f:
        f.write(f"{gen_dataset}, {gen_clip_model}, {gen_epoch},{gen_clip_loss}, {checkpoint_path},{dataset}, {model}, {attack_type}, {norm_type}, {epsilon}, {top1}, {top5}\n")
    print('done!')