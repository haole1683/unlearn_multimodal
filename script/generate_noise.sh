#!/bin/bash  

# "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/generator_best_epoch-200_loss-0.15745621855124767.pth",
GPATH=("./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/generator_best_epoch-500_loss-0.058220808281365666.pth")
TYPES=("sampleWise" "classWise")
DATASETS=("cifar10" "cifar100" "stl10")  


# 遍历backbone和dataset的所有组合  
for path in "${GPATH[@]}"; do  
    for type in "${TYPES[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            echo "Running experiment with path: $path and type: $type and dataset: $dataset"
            # 调用Python脚本并传递参数  
            python unlearn_stage2_generator_gen_noise.py --generator_path "$path" --noise_type "$type" --dataset "$dataset"
        done
    done  
done