import csv
import os
import time
import torch
import json

import logging

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)
    
def setup_logging(log_file, level):
    
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


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
    

def record_result_supervised(result, folder_path):
    import os
    # file_path = os.path.join(folder_path, "result.txt")
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    # with open(file_path, 'w') as f:
    #     for record in result:
    #         f.write(f'Epoch: {record["epoch"]}\n')
    #         f.write(f'Accuracy: {record["acc"]}\n')
    #         f.write(f'Class Accuracy: \n')
    #         for k,v in record['class_acc'].items():
    #             f.write(f'{k}: {v["correct_num"]},{v["total_num"]}, {v["correct_rate"]:.2f} | ')
    #         f.write('\n')
    
    # save as json
    file_path = os.path.join(folder_path, "result.json")
    with open(file_path, 'w') as f:
        json.dump(result, f)
        
    
class RecordSupervised(object):
    """record the supervised result

    Args:
        object (_type_): _description_
    """
    def __init__(self):
        self.result = []

    def add_one_record(self, epoch, acc_meter_train, loss_meter_train,acc_meter_test, loss_meter_test, class_correct_dict):
        record = {}
        record['epoch'] = epoch
        record['train_acc'] = acc_meter_train.avg
        record['train_loss'] = loss_meter_train.avg
        record['test_acc'] = acc_meter_test.avg
        record['test_loss'] = loss_meter_test.avg
        record['test_class_acc'] = class_correct_dict
        
        self.result.append(record)
        
    def add_one_record_value(self, epoch, acc_train, loss_train,acc_test, loss_test, class_correct_dict):
        record = {}
        record['epoch'] = epoch
        record['train_acc'] = acc_train
        record['train_loss'] = loss_train
        record['test_acc'] = acc_test
        record['test_loss'] = loss_test
        record['test_class_acc'] = class_correct_dict
        
        self.result.append(record)
        
    def save_result(self, path):
        # 按照每条记录的epoch的值排序
        self.result = sorted(self.result, key=lambda x: x['epoch'])
        record_result_supervised(self.result, path)
        print("Record saved! Path: ", path)
    
    def get_result(self):
        return self.result