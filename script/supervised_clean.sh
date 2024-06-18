#!/bin/bash  

# 定义backbone列表  
# BACKBONES=("ViT-B_16")  
BACKBONES=("resnet18" "resnet50" "resnet101" "ViT-B_16" "ViT-B_32")  
  
# 定义dataset列表  
DATASETS=("cifar10" "cifar100" "stl10")  

PRETRAIN=("True" "False")
# PRETRAIN=("True")
# PRETRAIN=("False")

# 遍历backbone和dataset的所有组合  
for BACKBONE in "${BACKBONES[@]}"; do  
    for DATASET in "${DATASETS[@]}"; do
        for PRETRAIN in "${PRETRAIN[@]}"; do  
            echo "Running experiment with backbone: $BACKBONE and dataset: $DATASET and pretrained: $PRETRAIN"  
            # 调用Python脚本并传递参数， 服了，参数写错了，全尼玛白跑了 ，傻逼东西，python会解析为字符串，没法传递这个参数。。。服了。。
            # https://blog.csdn.net/a15561415881/article/details/106088831
            if [ "$PRETRAIN" == "True" ]; then
                python unlearn_stage3_test_supervised.py --backbone "$BACKBONE" --dataset "$DATASET" --pretrained
            else
                python unlearn_stage3_test_supervised.py --backbone "$BACKBONE" --dataset "$DATASET"
            fi
        done
    done  
done