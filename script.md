# python unlearn_test_finetune_clip.py --device cuda:0 --finetune_dataset myLaion
# python unlearn_test_finetune_clip.py --device cuda:1 --poisoned --finetune_dataset myLaion
# python unlearn_test_finetune_clip.py --device cuda:2 --finetune_dataset cifar10
# python unlearn_test_finetune_clip.py --device cuda:3 --poisoned --finetune_dataset cifar10

# 2024-4-13
python unlearn_stage1_train_all_g.py --clip_model=both --device=cuda:0 --overwrite



# stage1 - train generator script 
Tip: each experiment cost about 2 days !!!
## For Single model RN50*4, the batch size need half(25G memory)
accelerate launch --config_file=accelerate_config.yaml unlearn_stage1_train_all_g_dis.py --clip_model=RN50x4 --trainset=all --batch_size=8 --overwrite
## For Single model version - RN50 - Doing
accelerate launch --config_file=accelerate_config.yaml unlearn_stage1_train_all_g_dis.py --clip_model=RN50 --trainset=all --batch_size=8 --overwrite 
## For Single model version - RN101 - Doing
accelerate launch --config_file=accelerate_config.yaml unlearn_stage1_train_all_g_dis.py --clip_model=RN101 --trainset=all --batch_size=8 --overwrite 
## For Single model version - ViT-B/16
accelerate launch --config_file=accelerate_config.yaml unlearn_stage1_train_all_g_dis.py --clip_model=ViT-B/16 --trainset=all --batch_size=8 --overwrite 
## For Single model version - ViT-B/32
accelerate launch --config_file=accelerate_config.yaml unlearn_stage1_train_all_g_dis.py --clip_model=ViT-B/32 --trainset=all --batch_size=8 --overwrite 
## For both model version(RN101 + ViT-B/16) - Done
✅ accelerate launch --config_file=accelerate_config.yaml unlearn_stage1_train_all_g_dis.py --clip_model=both --trainset=all --batch_size=32 --overwrite 
## For both model version(RN101 + ViT-B/32) 
accelerate launch --config_file=accelerate_config.yaml unlearn_stage1_train_all_g_dis.py --clip_model=both --trainset=all --batch_size=32 --overwrite 
## For Single model version - RN50 - Doing - with poison-natural image embedding loss
accelerate launch --config_file=accelerate_config.yaml unlearn_stage1_train_all_g_dis.py --clip_model=RN50 --trainset=all --batch_size=8 --overwrite 


# stage2 - generate noise from generator
# Note: update_z_fre set max, not update z
# text_prompt set fixed , use fixed text prompt
# gen1 : generate noise for supervised model
# gen2 : generate noise for myLaion dataset

## RN50-Cifar10-sampleWise
✅ python unlearn_stage2_generator_gen_noise.py --generator_path="output/unlearn_stage1_train_g_unlearn/gen_all-RN50/checkpoint/generator_best_epoch-315_loss-0.7207418711045938.pth" --device='cuda:0' --clip_model=RN50 --gen_which=gen1 --overwrite --noise_type=sampleWise --dataset=cifar10
## RN50-Cifar10-classWise
✅ python unlearn_stage2_generator_gen_noise.py --generator_path="output/unlearn_stage1_train_g_unlearn/gen_all-RN50/checkpoint/generator_best_epoch-315_loss-0.7207418711045938.pth" --device='cuda:1' --clip_model=RN50 --gen_which=gen1 --overwrite --noise_type=classWise --dataset=cifar10
## RN50-Cifar100-sampleWise
✅ python unlearn_stage2_generator_gen_noise.py --generator_path="output/unlearn_stage1_train_g_unlearn/gen_all-RN50/checkpoint/generator_best_epoch-315_loss-0.7207418711045938.pth" --device='cuda:2' --clip_model=RN50 --gen_which=gen1 --overwrite --noise_type=sampleWise --dataset=cifar100
## RN50-Cifar100-classWise
✅ python unlearn_stage2_generator_gen_noise.py --generator_path="output/unlearn_stage1_train_g_unlearn/gen_all-RN50/checkpoint/generator_best_epoch-315_loss-0.7207418711045938.pth" --device='cuda:3' --clip_model=RN50 --gen_which=gen1 --overwrite --noise_type=classWise --dataset=cifar100
## RN50-STL10-sampleWise
✅ python unlearn_stage2_generator_gen_noise.py --generator_path="output/unlearn_stage1_train_g_unlearn/gen_all-RN50/checkpoint/generator_best_epoch-315_loss-0.7207418711045938.pth" --device='cuda:2' --clip_model=RN50 --gen_which=gen1 --overwrite --noise_type=sampleWise --dataset=stl10
## RN50-STL10-classWise
✅ python unlearn_stage2_generator_gen_noise.py --generator_path="output/unlearn_stage1_train_g_unlearn/gen_all-RN50/checkpoint/generator_best_epoch-315_loss-0.7207418711045938.pth" --device='cuda:3' --clip_model=RN50 --gen_which=gen1 --overwrite --noise_type=classWise --dataset=stl10


## RN101-Cifar10-sampleWise
✅ python unlearn_stage2_generator_gen_noise.py --generator_path="./output/unlearn_stage1_train_g_unlearn/gen_all-ViT-B_16/checkpoint/generator_best_epoch-235_loss-0.03984706313345276.pth" --device='cuda:0' --clip_model=ViT-B/16 --gen_which=gen1 --overwrite --noise_type=sampleWise --dataset=cifar10
## RN101-Cifar10-classWise
✅ python unlearn_stage2_generator_gen_noise.py --generator_path="./output/unlearn_stage1_train_g_unlearn/gen_all-ViT-B_16/checkpoint/generator_best_epoch-235_loss-0.03984706313345276.pth" --device='cuda:0' --clip_model=ViT-B/16 --gen_which=gen1 --overwrite --noise_type=classWise --dataset=cifar10
## RN101-Cifar100-sampleWise
✅ python unlearn_stage2_generator_gen_noise.py --generator_path="./output/unlearn_stage1_train_g_unlearn/gen_all-ViT-B_16/checkpoint/generator_best_epoch-235_loss-0.03984706313345276.pth" --device='cuda:0' --clip_model=ViT-B/16 --gen_which=gen1 --overwrite --noise_type=sampleWise --dataset=cifar100
## RN101-Cifar100-classWise
✅ python unlearn_stage2_generator_gen_noise.py --generator_path="./output/unlearn_stage1_train_g_unlearn/gen_all-ViT-B_16/checkpoint/generator_best_epoch-235_loss-0.03984706313345276.pth" --device='cuda:0' --clip_model=ViT-B/16 --gen_which=gen1 --overwrite --noise_type=classWise --dataset=cifar100
## RN101-stl10-sampleWise
✅ python unlearn_stage2_generator_gen_noise.py --generator_path="./output/unlearn_stage1_train_g_unlearn/gen_all-ViT-B_16/checkpoint/generator_best_epoch-235_loss-0.03984706313345276.pth" --device='cuda:0' --clip_model=ViT-B/16 --gen_which=gen1 --overwrite --noise_type=sampleWise --dataset=stl10
## RN101-stl10-classWise
✅ python unlearn_stage2_generator_gen_noise.py --generator_path="./output/unlearn_stage1_train_g_unlearn/gen_all-ViT-B_16/checkpoint/generator_best_epoch-235_loss-0.03984706313345276.pth" --device='cuda:0' --clip_model=ViT-B/16 --gen_which=gen1 --overwrite --noise_type=classWise --dataset=stl10


## ViT-B_16-Cifar10-sampleWise
python unlearn_stage2_generator_gen_noise.py --generator_path="./output/unlearn_stage1_train_g_unlearn/gen_all-ViT-B_16/checkpoint/generator_best_epoch-235_loss-0.03984706313345276.pth" --device='cuda:0' --clip_model=ViT-B/16 --gen_which=gen1 --overwrite --noise_type=sampleWise --dataset=cifar10
## ViT-B_16-Cifar10-classWise
python unlearn_stage2_generator_gen_noise.py --generator_path="./output/unlearn_stage1_train_g_unlearn/gen_all-ViT-B_16/checkpoint/generator_best_epoch-235_loss-0.03984706313345276.pth" --device='cuda:0' --clip_model=ViT-B/16 --gen_which=gen1 --overwrite --noise_type=classWise --dataset=cifar10
## ViT-B_16-Cifar100-sampleWise
python unlearn_stage2_generator_gen_noise.py --generator_path="./output/unlearn_stage1_train_g_unlearn/gen_all-ViT-B_16/checkpoint/generator_best_epoch-235_loss-0.03984706313345276.pth" --device='cuda:0' --clip_model=ViT-B/16 --gen_which=gen1 --overwrite --noise_type=sampleWise --dataset=cifar100
## ViT-B_16-Cifar100-classWise
python unlearn_stage2_generator_gen_noise.py --generator_path="./output/unlearn_stage1_train_g_unlearn/gen_all-ViT-B_16/checkpoint/generator_best_epoch-235_loss-0.03984706313345276.pth" --device='cuda:0' --clip_model=ViT-B/16 --gen_which=gen1 --overwrite --noise_type=classWise --dataset=cifar100
## ViT-B_16-stl10-sampleWise
python unlearn_stage2_generator_gen_noise.py --generator_path="./output/unlearn_stage1_train_g_unlearn/gen_all-ViT-B_16/checkpoint/generator_best_epoch-235_loss-0.03984706313345276.pth" --device='cuda:0' --clip_model=ViT-B/16 --gen_which=gen1 --overwrite --noise_type=sampleWise --dataset=stl10
## ViT-B_16-stl10-classWise
python unlearn_stage2_generator_gen_noise.py --generator_path="./output/unlearn_stage1_train_g_unlearn/gen_all-ViT-B_16/checkpoint/generator_best_epoch-235_loss-0.03984706313345276.pth" --device='cuda:0' --clip_model=ViT-B/16 --gen_which=gen1 --overwrite --noise_type=classWise --dataset=stl10


# stage3 - test noise script
## image-text-pair training for finetune clip model
### √ For from pretrained 
#### √ RN50
##### √ not poisoned
√ python unlearn_stage3_test_finetune_clip.py --device="cuda:0" --clip_model=RN50 --batch_size=256
##### poisoned
√ python unlearn_stage3_test_finetune_clip.py --device="cuda:1" --clip_model=RN50 --batch_size=256 --poisoned --noise_path="./output/unlearn_stage2_generate_noise/RN50/noise_gen2_46221-224-224_all_RN50.pt"
### For from scratch 
#### RN50
##### not poisoned
√ python unlearn_stage3_test_finetune_clip.py --device="cuda:2" --clip_model=RN50 --batch_size=256 --from_scratch 
##### poisoned
√ python unlearn_stage3_test_finetune_clip.py --device="cuda:2" --clip_model=RN50 --batch_size=256 --from_scratch --poisoned --noise_path="./output/unlearn_stage2_generate_noise/RN50/noise_gen2_46221-224-224_all_RN50.pt"
#### ViT-B/16
##### not poisoned
√ python unlearn_stage3_test_finetune_clip.py --device="cuda:0" --clip_model=ViT-B/16 --batch_size=128 --from_scratch 
##### poisoned
√ python unlearn_stage3_test_finetune_clip.py --device="cuda:1" --clip_model=ViT-B/16 --batch_size=128 --from_scratch --poisoned --noise_path="./output/unlearn_stage2_generate_noise/ViT-B-16/noise_gen2_46221-224-224_all_ViT-B-16.pt"
#### both
##### not poisoned
√ python unlearn_stage3_test_finetune_clip.py --device="cuda:0" --clip_model=ViT-B/16 --batch_size=128 --from_scratch 
##### poisoned
###### model - ViT-B/16
△ python unlearn_stage3_test_finetune_clip.py --device="cuda:1" --clip_model=ViT-B/16 --batch_size=128 --from_scratch --poisoned --noise_path="./output/unlearn_stage2_generate_noise/both-encoder-ViT-B-16/noise_gen2_46221-224-224_all_ViT-B-16.pt"
###### model - RN101
△ python unlearn_stage3_test_finetune_clip.py --device="cuda:2" --clip_model=RN101 --batch_size=128 --from_scratch --poisoned --noise_path="./output/unlearn_stage2_generate_noise/both-encoder-RN101/noise_gen2_46221-224-224_all_RN101.pt"
###### model - RN50
△ python unlearn_stage3_test_finetune_clip.py --device="cuda:3" --clip_model=RN50 --batch_size=128 --from_scratch --poisoned --noise_path="./output/unlearn_stage2_generate_noise/both-encoder-RN101/noise_gen2_46221-224-224_all_RN101.pt"
###### model - ViT-B/32
△ python unlearn_stage3_test_finetune_clip.py --device="cuda:3" --clip_model=ViT-B/32 --batch_size=128 --from_scratch --poisoned --noise_path="./output/unlearn_stage2_generate_noise/both-encoder-ViT-B-16/noise_gen2_46221-224-224_all_ViT-B-16.pt"

## self-supervised model
python unlearn_stage3_test_self_supervised.py --dataset='cifar10' --device='cuda:0'
python unlearn_stage3_test_self_supervised.py --dataset='cifar10' --device='cuda:1' --poisoned --noise_path="./output/unlearn_stage2_generate_noise/ViT-B-16/cifar10/noise_gen1_ViT-B-16_cifar10_all.pt"
python unlearn_stage3_test_self_supervised.py --dataset='stl10' --device='cuda:2'
python unlearn_stage3_test_self_supervised.py --dataset='stl10' --device='cuda:3' --poisoned --noise_path="./output/unlearn_stage2_generate_noise/ViT-B-16/stl10/noise_gen1_ViT-B-16_stl10_all.pt"
## supervised model 
./${output_dir}/${dataset}/${natural/poisoned}/${pretrained/scratch}${poisoned_source}

### cifar10-natural-pretrained
✅ python unlearn_stage3_test_supervised.py --dataset='cifar10' --device='cuda:0' --pretrained 
### cifar10-natural-scratch
✅ python unlearn_stage3_test_supervised.py --dataset='cifar10' --device='cuda:1' 

### cifar100-natural-pretrained
✅ python unlearn_stage3_test_supervised.py --dataset='cifar100' --device='cuda:0' --pretrained 
### cifar100-natural-scratch
✅ python unlearn_stage3_test_supervised.py --dataset='cifar100' --device='cuda:1' 

### stl10-natural-pretrained
✅ python unlearn_stage3_test_supervised.py --dataset='stl10' --device='cuda:2' --pretrained 
### stl10-natural-scratch
✅ python unlearn_stage3_test_supervised.py --dataset='stl10' --device='cuda:3' 


### cifar10-poisoned-pretrain-RN50_Noise_sampleWise
✅ python unlearn_stage3_test_supervised.py --dataset='cifar10' --device='cuda:0' --pretrained --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/RN50/cifar10/sampleWise/noise_gen1_RN50_cifar10_all.pt" 
### cifar10-poisoned-pretrain-RN50_Noise_classWise
✅ python unlearn_stage3_test_supervised.py --dataset='cifar10' --device='cuda:0' --pretrained --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/RN50/cifar10/classWise/noise_gen1_RN50_cifar10_all.pt" 
### cifar10-poisoned-scratch-RN101_Noise_sampleWise
✅ python unlearn_stage3_test_supervised.py --dataset='cifar10' --device='cuda:1' --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/RN50/cifar10/sampleWise/noise_gen1_RN50_cifar10_all.pt" 
### cifar10-poisoned-scratch-RN101_Noise_classWise
✅ python unlearn_stage3_test_supervised.py --dataset='cifar10' --device='cuda:2' --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/RN50/cifar10/classWise/noise_gen1_RN50_cifar10_all.pt" 
### cifar10-poisoned-pretrain-ViT-B_16_Noise_sampleWise
✅ python unlearn_stage3_test_supervised.py --dataset='cifar10' --device='cuda:0' --pretrained --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/ViT-B_16/cifar10/sampleWise/noise_gen1_ViT-B-16_cifar10_all.pt" 
### cifar10-poisoned-pretrain-ViT_B_16_Noise_classWise
✅ python unlearn_stage3_test_supervised.py --dataset='cifar10' --device='cuda:0' --pretrained --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/ViT-B_16/cifar10/classWise/noise_gen1_ViT-B-16_cifar10_all.pt" 
### cifar10-poisoned-scratch-ViT_B_32_Noise_sampleWise
✅ python unlearn_stage3_test_supervised.py --dataset='cifar10' --device='cuda:1' --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/ViT-B_16/cifar10/sampleWise/noise_gen1_ViT-B-16_cifar10_all.pt" 
### cifar10-poisoned-scratch-ViT_B_32_Noise_classWise
✅ python unlearn_stage3_test_supervised.py --dataset='cifar10' --device='cuda:2' --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/ViT-B_16/cifar10/classWise/noise_gen1_ViT-B-16_cifar10_all.pt" 

### cifar100-poisoned-pretrain-RN50_Noise_sampleWise
✅ python unlearn_stage3_test_supervised.py --dataset='cifar100' --device='cuda:0' --pretrained --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/RN50/cifar100/sampleWise/noise_gen1_RN50_cifar100_all.pt" 
### cifar100-poisoned-pretrain-RN50_Noise_classWise
✅ python unlearn_stage3_test_supervised.py --dataset='cifar100' --device='cuda:3' --pretrained --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/RN50/cifar100/classWise/noise_gen1_RN50_cifar100_all.pt" 
### cifar100-poisoned-scratch-RN101_Noise_sampleWise
✅ python unlearn_stage3_test_supervised.py --dataset='cifar100' --device='cuda:1' --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/RN50/cifar100/sampleWise/noise_gen1_RN50_cifar100_all.pt" 
### cifar100-poisoned-scratch-RN101_Noise_classWise
✅ python unlearn_stage3_test_supervised.py --dataset='cifar100' --device='cuda:2' --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/RN50/cifar100/classWise/noise_gen1_RN50_cifar100_all.pt" 
### cifar100-poisoned-pretrain-ViT-B_16_Noise_sampleWise
✅ python unlearn_stage3_test_supervised.py --dataset='cifar100' --device='cuda:0' --pretrained --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/ViT-B_16/cifar100/sampleWise/noise_gen1_ViT-B-16_cifar100_all.pt" 
### cifar100-poisoned-pretrain-ViT_B_16_Noise_classWise
✅ python unlearn_stage3_test_supervised.py --dataset='cifar100' --device='cuda:1' --pretrained --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/ViT-B_16/cifar100/classWise/noise_gen1_ViT-B-16_cifar100_all.pt" 
### cifar100-poisoned-scratch-ViT_B_16_Noise_sampleWise
✅ python unlearn_stage3_test_supervised.py --dataset='cifar100' --device='cuda:2' --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/ViT-B_16/cifar100/sampleWise/noise_gen1_ViT-B-16_cifar100_all.pt" 
### cifar100-poisoned-scratch-ViT_B_16_Noise_classWise
✅ python unlearn_stage3_test_supervised.py --dataset='cifar100' --device='cuda:3' --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/ViT-B_16/cifar100/classWise/noise_gen1_ViT-B-16_cifar100_all.pt" 


### stl10-poisoned-pretrain-RN50_Noise_sampleWise
✅ python unlearn_stage3_test_supervised.py --dataset='stl10' --device='cuda:0' --pretrained --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/RN50/stl10/sampleWise/noise_gen1_RN50_stl10_all.pt" 
### stl10-poisoned-pretrain-RN50_Noise_classWise
✅ python unlearn_stage3_test_supervised.py --dataset='stl10' --device='cuda:1' --pretrained --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/RN50/stl10/classWise/noise_gen1_RN50_stl10_all.pt" 
### stl10-poisoned-scratch-RN101_Noise_sampleWise
✅ python unlearn_stage3_test_supervised.py --dataset='stl10' --device='cuda:2' --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/RN50/stl10/sampleWise/noise_gen1_RN50_stl10_all.pt" 
### stl10-poisoned-scratch-RN101_Noise_classWise
✅ python unlearn_stage3_test_supervised.py --dataset='stl10' --device='cuda:3' --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/RN50/stl10/classWise/noise_gen1_RN50_stl10_all.pt" 
### stl10-poisoned-pretrain-ViT-B_16_Noise_sampleWise
✅ python unlearn_stage3_test_supervised.py --dataset='stl10' --device='cuda:0' --pretrained --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/ViT-B_16/stl10/sampleWise/noise_gen1_ViT-B-16_stl10_all.pt" 
### stl10-poisoned-pretrain-ViT_B_16_Noise_classWise
✅ python unlearn_stage3_test_supervised.py --dataset='stl10' --device='cuda:1' --pretrained --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/ViT-B_16/stl10/classWise/noise_gen1_ViT-B-16_stl10_all.pt" 
### stl10-poisoned-scratch-ViT_B_32_Noise_sampleWise
✅ python unlearn_stage3_test_supervised.py --dataset='stl10' --device='cuda:2' --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/ViT-B_16/stl10/sampleWise/noise_gen1_ViT-B-16_stl10_all.pt" 
### stl10-poisoned-scratch-ViT_B_32_Noise_classWise
✅ python unlearn_stage3_test_supervised.py --dataset='stl10' --device='cuda:3' --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/ViT-B_16/stl10/classWise/noise_gen1_ViT-B-16_stl10_all.pt" 








## 测试一下到底CLIP能不能from_scratch训练。。。调整一下学习率？？
python unlearn_stage3_test_finetune_clip.py --device="cuda:3" --clip_model=RN50 --batch_size=256 --output_dir="./output/unlearn_temp_test2/" --from_scratch --lr="1e-3"

Conclusion : 不行，Test accuary死活上不去。。。

## 测试一下sample-wise（random z freq AND not fix the text_propt_straegy）的noise poison cifar-10的效果
python unlearn_stage2_generator_gen_noise.py --generator_path="./output/unlearn_stage1_train_g_unlearn/gen_all-ViT-B_16/checkpoint/generator_best_epoch-235_loss-0.03984706313345276.pth" --device='cuda:2' --clip_model=ViT-B/16 --overwrite --update_z_freq=10 --text_prompt_stragegy=random --output_dir="./output/unlearn_test_sample-wise_stage2_generate_noise/" --gen_which=gen1

python unlearn_stage3_test_supervised.py --dataset='cifar10' --device='cuda:0' --noise_path="./output/unlearn_test_sample-wise_stage2_generate_noise/B_16/cifar10/noise_gen1_ViT-B-16_cifar10_all.pt" --output_dir="./output/unlearn_stage3_sample-wise_test_self_supervised" --poisoned

python unlearn_stage3_test_supervised.py --dataset='stl10' --device='cuda:0' --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_test_sample-wise_stage2_generate_noise/B_16/stl10/noise_gen1_ViT-B-16_stl10_all.pt" --output_dir="./output/unlearn_stage3_sample-wise_test_self_supervised" --poisoned

效果还凑合。。
sample-wise epilision=16 attack for cifar-10 有下降

## 测试一下cifar-100 supervised learning

### normal training
python unlearn_stage3_test_supervised.py --dataset='cifar100' --device='cuda:0' 
### poison training
#### generate noise 
python unlearn_stage2_generator_gen_noise.py --generator_path="./output/unlearn_stage1_train_g_unlearn/gen_all-ViT-B_16/checkpoint/generator_best_epoch-235_loss-0.03984706313345276.pth" --device='cuda:3' --clip_model=ViT-B/16 --overwrite --update_z_freq=10 --text_prompt_stragegy=random --output_dir="./output/unlearn_test_sample-wise_stage2_generate_noise/cifar100" --gen_which=gen1 --dataset=cifar100
#### poison trianing
python unlearn_stage3_test_supervised.py --dataset='cifar100' --device='cuda:0' --poisoned --noise_path="./output/unlearn_test_sample-wise_stage2_generate_noise/B_16/cifar100/noise_gen1_ViT-B-16_cifar100_all.pt"








train clip from scratch on coco dataset

python unlearn_stage3_test_finetune_clip_coco.py --device="cuda:0" --clip_model=RN50 --batch_size=8 --output_dir="./output/unlearn_train_from_scratch_on_coco/" --from_scratch --lr="1e-3" --finetune_dataset=myLaion


















Group 1
2 - python unlearn_stage3_test_finetune_clip_adapter.py --device='cuda:0' --clip_model=RN50 
3 - python unlearn_stage3_test_finetune_clip_adapter.py --device='cuda:1' --clip_model=RN50 --poisoned --noise_path="./output/unlearn_stage2_generate_noise/RN50/noise_gen2_46221-224-224_all_RN50.pt" 
4 - python unlearn_stage3_test_finetune_clip_adapter.py --device='cuda:2' --clip_model=ViT-B/16 
5 - python unlearn_stage3_test_finetune_clip_adapter.py --device='cuda:3' --clip_model=ViT-B/16 --poisoned --noise_path="./output/unlearn_stage2_generate_noise/ViT-B_16/all/classWise/noise_gen2_46221-224-224_all_ViT-B-16.pt"
