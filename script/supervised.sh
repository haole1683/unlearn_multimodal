#!/bin/bash  
  
# 定义backbone列表  
BACKBONES=("resnet18" "resnet50" "resnet101" "vgg16" "mobilenet_v2")  
  
# 定义dataset列表  
DATASETS=("cifar10" "cifar100" "stl10")  
  
# 遍历backbone和dataset的所有组合  
for BACKBONE in "${BACKBONES[@]}"; do  
    for DATASET in "${DATASETS[@]}"; do  
        echo "Running experiment with backbone: $BACKBONE and dataset: $DATASET"  
        # 调用Python脚本并传递参数  
        python unlearn_stage3_test_supervised.py --backbone "$BACKBONE" --dataset "$DATASET"  
  
        # 如果需要，可以在这里添加逻辑来处理实验的输出或日志  
        # 例如，将输出重定向到不同的文件  
        # python run_experiment.py --backbone "$BACKBONE" --dataset "$DATASET" > "$BACKBONE-$DATASET.log" 2>&1  
    done  
done