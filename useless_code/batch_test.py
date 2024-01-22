
import os
import sys
import torch

checkpoint = [
    "/remote-home/songtianwei/research/unlearn_multimodal/output/train_generator_max_loss/checkpoint_epoch_10.pth",
]
available_device = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
clip_model = []
seed = 42
test_method = ['clean', 'generator', 'random']
norm_type = ['l2', 'linf']
epsilon = [16, 8 ,4]
batch_size = 64
dataset = ['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet']

python_script = "python test_attack.py --checkpoint {} --device {} --seed {} --batch_size {} --clip_model {} --test_method {} --norm_type {} --epsilon {} --dataset {}"




def execute():
    for c in checkpoint:
        for t in test_method:
            for n in norm_type:
                for e in epsilon:
                    for ds in dataset:
                        d = available_device[0]
                        cmd = python_script.format(c, d, seed, batch_size, clip_model, t, n, e, ds)
                        print(cmd)
                        os.system(cmd)

if __name__ == "__main__":
    execute()