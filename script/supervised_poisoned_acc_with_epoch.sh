#!/bin/bash  

PREFIX_PATH="./output/unlearn_stage2_generate_noise_acc_with_epoch"

# 定义backbone列表  
BACKBONES=(
    "resnet50"
#  "ViT-B_16" 
#  "ViT-B_32"
)  

# 定义dataset列表  
DATASETS=(
# "cifar10"
"cifar100"
)  

# 是否为预训练模型
PRETRAIN=(
    # "True" 
    "False"
)

# Generator版本
# G_CLIP_VERSION=('ViT-B-16')
G_CLIP_VERSION=('ViT-B-32')

# Noise Type
NOISE_TYPE=('classWise')

# Epoch Num
# EPOCH_NUM1=(0 50 100 150 200 250 300)
# EPOCH_NUM2=(0 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340 360 380 400 420 440 460 480 500)
EPOCH_NUM3=(360 380 400 420 440 460 480 500)
EPOCH_NUM=("${EPOCH_NUM3[@]}")
# 拼接卢杰

# 遍历backbone和dataset的所有组合  
for G_CLIP in "${G_CLIP_VERSION[@]}"; do
    for EPOCH in "${EPOCH_NUM[@]}"; do
        for BACKBONE in "${BACKBONES[@]}"; do  
            for DATASET in "${DATASETS[@]}"; do
                for PRETRAIN in "${PRETRAIN[@]}"; do  
                    for TYPE in "${NOISE_TYPE[@]}"; do
                        # output/unlearn_stage2_generate_noise_acc_with_epoch/ViT-B-16/epoch-0/cifar10/classWise/noise_gen1_ViT-B-16_cifar10_all.pt
                        the_noise_path="${PREFIX_PATH}/${G_CLIP}/epoch-${EPOCH}/${DATASET}/${TYPE}/noise_gen1_${G_CLIP}_${DATASET}_all.pt"
                        echo "Running poisoned experiment with backbone: $BACKBONE and dataset: $DATASET and pretrain: $PRETRAIN and G_CLIP: $G_CLIP and NOISE_PATH: $the_noise_path"  
                        # 调用Python脚本并传递参数  
                        if [ "$PRETRAIN" == "True" ]; then
                            echo "Pretrain"
                            python unlearn_stage3_test_supervised.py --backbone "$BACKBONE" --dataset "$DATASET" --poisoned --noise_path "$the_noise_path" --pretrain --record_epoch "$EPOCH" --output_dir ./output/unlearn_stage3_test_supervised_acc_with_epoch/
                        else
                            echo "No Pretrain"
                            python unlearn_stage3_test_supervised.py --backbone "$BACKBONE" --dataset "$DATASET" --poisoned --noise_path "$the_noise_path" --record_epoch "$EPOCH" --output_dir ./output/unlearn_stage3_test_supervised_acc_with_epoch/
                        fi
                    done
                done
            done
        done  
    done
done