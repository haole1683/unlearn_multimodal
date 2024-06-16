#!/bin/bash  

# 定义backbone列表  
BACKBONES=("resnet18" "resnet50" "resnet101" "ViT-B_16" "ViT-B_32")  
  
# 定义dataset列表  
DATASETS=("cifar10" "cifar100" "stl10")  

PRETRAIN=("True" "False")

# 遍历backbone和dataset的所有组合  
for BACKBONE in "${BACKBONES[@]}"; do  
    for DATASET in "${DATASETS[@]}"; do
        for PRETRAIN in "${PRETRAIN[@]}"; do  
            echo "Running experiment with backbone: $BACKBONE and dataset: $DATASET and pretrain: $PRETRAIN"  
            # 调用Python脚本并传递参数  
            python unlearn_stage3_test_supervised.py --backbone "$BACKBONE" --dataset "$DATASET" --pretrain "$PRETRAIN"
        done
    done  
done