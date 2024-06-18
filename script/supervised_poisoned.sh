#!/bin/bash  

PREFIX_PATH="./output/unlearn_stage2_generate_noise_temp1"

# 定义backbone列表  
BACKBONES=("resnet18" "resnet50" "resnet101" "ViT-B_16" "ViT-B_32")  
  
# 定义dataset列表  
DATASETS=("cifar10" "cifar100" "stl10")  

# 是否为预训练模型
PRETRAIN=("True" "False")

# Generator版本
# G_CLIP_VERSION=('RN50'  'RN101' 'RN50x4' 'ViT-B-32' 'ViT-B-16')
G_CLIP_VERSION=('ViT-B-32')

# Noise Type
NOISE_TYPE=('classWise' 'sampleWise')

# 拼接卢杰

# 遍历backbone和dataset的所有组合  
for BACKBONE in "${BACKBONES[@]}"; do  
    for DATASET in "${DATASETS[@]}"; do
        for PRETRAIN in "${PRETRAIN[@]}"; do  
            for G_CLIP in "${G_CLIP_VERSION[@]}"; do
                for TYPE in "${NOISE_TYPE[@]}"; do
                    the_noise_path="${PREFIX_PATH}/${G_CLIP}/${DATASET}/${TYPE}/noise_gen1_${G_CLIP}_${DATASET}_all.pt"
                    echo "Running poisoned experiment with backbone: $BACKBONE and dataset: $DATASET and pretrain: $PRETRAIN and G_CLIP: $G_CLIP and NOISE_PATH: $the_noise_path"  
                    # 调用Python脚本并传递参数  
                    if [ "$PRETRAIN" == "True" ]; then
                        python unlearn_stage3_test_supervised.py --backbone "$BACKBONE" --dataset "$DATASET" --poisoned --noise_path "$the_noise_path" --pretrain
                    else
                        python unlearn_stage3_test_supervised.py --backbone "$BACKBONE" --dataset "$DATASET" --poisoned --noise_path "$the_noise_path" 
                    fi
                done
            done
        done
    done  
done