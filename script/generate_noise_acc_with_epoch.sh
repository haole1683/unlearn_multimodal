#!/bin/bash  


# Step1: Generate Noise
GPATH_VIT=(
    # ViT-B/16
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-16/checkpoint/generator_best_epoch-0_loss-0.8576622623385806.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-16/checkpoint/generator_best_epoch-50_loss-0.7174993714728912.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-16/checkpoint/generator_best_epoch-100_loss-0.5994411492115491.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-16/checkpoint/generator_best_epoch-150_loss-0.3760107533756273.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-16/checkpoint/generator_best_epoch-200_loss-0.2129581318079651.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-16/checkpoint/generator_best_epoch-250_loss-0.14782940225425736.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-16/checkpoint/generator_best_epoch-300_loss-0.11573614112355492.pth"

    # ViT-B/32
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-0_loss-1.2603146454472562.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-20_loss-1.2052605202445736.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-40_loss-1.186753110839175.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-60_loss-1.017702035573654.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-80_loss-0.7942867370652946.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-100_loss-0.5891858519284756.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-120_loss-0.42839870714780054.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-140_loss-0.3037150747332222.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-160_loss-0.22932301019822365.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-180_loss-0.1846248187705294.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-200_loss-0.15745621855124767.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-220_loss-0.13612845939991278.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-240_loss-0.1236413200728563.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-260_loss-0.1117707160160397.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-280_loss-0.10175571809986453.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-300_loss-0.09485640283270831.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-320_loss-0.08935104081512013.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-340_loss-0.08368754915054484.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-360_loss-0.07928085946417474.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-380_loss-0.07453714274837599.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-400_loss-0.0704150368898978.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-420_loss-0.06865864697021323.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-440_loss-0.06511809693818743.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-460_loss-0.06265147104536817.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-480_loss-0.060398892043924436.pth"
    # "./output/unlearn_stage1_train_g_unlearn/gen-coco-ViT-B-32/checkpoint/generator_best_epoch-500_loss-0.058220808281365666.pth"
)

TYPES=(
    # "sampleWise" 
    "classWise"
)
DATASETS=(
    "cifar10" 
    "cifar100" 
    # "stl10"
)  

# echo "${GPATH_VIT[@]}"
# echo "${TYPES[@]}"
# echo "${DATASETS[@]}"


# 遍历backbone和dataset的所有组合  
for path in "${GPATH_VIT[@]}"; do  
    for type in "${TYPES[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            echo "PATH: $path"
            echo "Running experiment with path: $path and type: $type and dataset: $dataset"
            # 调用Python脚本并传递参数  
            python unlearn_stage2_generator_gen_noise.py --record_epoch T --output_dir ./output/unlearn_stage2_generate_noise_acc_with_epoch/ --generator_path "$path" --noise_type "$type" --dataset "$dataset"  
        done
    done  
done