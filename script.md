# python unlearn_test_finetune_clip.py --device cuda:0 --finetune_dataset myLaion
# python unlearn_test_finetune_clip.py --device cuda:1 --poisoned --finetune_dataset myLaion
# python unlearn_test_finetune_clip.py --device cuda:2 --finetune_dataset cifar10
# python unlearn_test_finetune_clip.py --device cuda:3 --poisoned --finetune_dataset cifar10

# 2024-4-13
python unlearn_stage1_train_all_g.py --clip_model=both --device=cuda:0 --overwrite



# stage1 - train generator script 
## For Single model RN50*4, the batch size need half(25G memory)
accelerate launch --config_file=accelerate_config.yaml unlearn_stage1_train_all_g_dis.py --clip_model=RN50x4 --trainset=all --batch_size=16 --overwrite
## For Single model version - RN50 - Doing
accelerate launch --config_file=accelerate_config.yaml unlearn_stage1_train_all_g_dis.py --clip_model=RN50 --trainset=all --batch_size=8 --overwrite 
## For Single model version - RN101 - Doing
accelerate launch --config_file=accelerate_config.yaml unlearn_stage1_train_all_g_dis.py --clip_model=RN101 --trainset=all --batch_size=8 --overwrite 
## For Single model version - ViT-B/16
accelerate launch --config_file=accelerate_config.yaml unlearn_stage1_train_all_g_dis.py --clip_model=ViT-B/16 --trainset=all --batch_size=8 --overwrite 
## For Single model version - ViT-B/32
accelerate launch --config_file=accelerate_config.yaml unlearn_stage1_train_all_g_dis.py --clip_model=ViT-B/32 --trainset=all --batch_size=8 --overwrite 
## For both model version(RN101 + ViT-B/16) - Done
accelerate launch --config_file=accelerate_config.yaml unlearn_stage1_train_all_g_dis.py --clip_model=both --trainset=all --batch_size=32 --overwrite 
## For both model version(RN101 + ViT-B/32) 
accelerate launch --config_file=accelerate_config.yaml unlearn_stage1_train_all_g_dis.py --clip_model=both --trainset=all --batch_size=32 --overwrite 

# stage2 - generate noise from generator
# Note: update_z_fre set max, not update z
# text_prompt set fixed , use fixed text prompt
# gen1 : generate noise for supervised model
# gen2 : generate noise for myLaion dataset
python unlearn_stage2_generator_gen_noise.py --generator_path="output/unlearn_stage1_train_g_unlearn/gen_all-both/checkpoint/generator_best_epoch-214_loss-0.11523310208746033.pth" --device='cuda:0' --clip_model=ViT-B_16 --gen_which=all --overwrite --update_z_freq=1000000 --text_prompt_stragegy=fixed

python unlearn_stage2_generator_gen_noise.py --generator_path="./output/unlearn_stage1_train_g_unlearn/gen_all-ViT-B_16/checkpoint/generator_best_epoch-235_loss-0.03984706313345276.pth" --device='cuda:2' --clip_model=ViT-B/16 --gen_which=all --overwrite --update_z_freq=1000000 --text_prompt_stragegy=fixed


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
python unlearn_stage3_test_supervised.py --dataset='cifar10' --device='cuda:0' --pretrained 
### cifar10-natural-scratch
python unlearn_stage3_test_supervised.py --dataset='cifar10' --device='cuda:1' 
### cifar10-poisoned-pretrain-RN50_Noise_sampleWise
python unlearn_stage3_test_supervised.py --dataset='cifar10' --device='cuda:0' --pretrained 
### cifar10-poisoned-pretrain-RN50_Noise_classWise
python unlearn_stage3_test_supervised.py --dataset='cifar10' --device='cuda:0' --pretrained 
### cifar10-poisoned-pretrain-RN101_Noise_sampleWise
python unlearn_stage3_test_supervised.py --dataset='cifar10' --device='cuda:0' --pretrained  
### cifar10-poisoned-pretrain-RN101_Noise_classWise
python unlearn_stage3_test_supervised.py --dataset='cifar10' --device='cuda:0' --pretrained  
### cifar10-poisoned-pretrain-ViT_B_16_Noise_sampleWise
python unlearn_stage3_test_supervised.py --dataset='cifar10' --pretrained --device='cuda:0' 
### cifar10-poisoned-pretrain-ViT_B_16_Noise_classWise
python unlearn_stage3_test_supervised.py --dataset='cifar10' --pretrained --device='cuda:0' 
### cifar10-poisoned-pretrain-ViT_B_32_Noise_sasmpleWise
python unlearn_stage3_test_supervised.py --dataset='cifar10' --pretrained --device='cuda:0' 
### cifar10-poisoned-pretrain-ViT_B_32_Noise_classWise
python unlearn_stage3_test_supervised.py --dataset='cifar10' --pretrained --device='cuda:0' 
### cifar10-poisoned-scratch-RN50_Noise_sampleWise
python unlearn_stage3_test_supervised.py --dataset='cifar10' --device='cuda:0' --pretrained 
### cifar10-poisoned-scratch-RN50_Noise_classWise
python unlearn_stage3_test_supervised.py --dataset='cifar10' --device='cuda:0' --pretrained 
### cifar10-poisoned-scratch-RN101_Noise_sampleWise
python unlearn_stage3_test_supervised.py --dataset='cifar10' --pretrained --device='cuda:0' 
### cifar10-poisoned-scratch-RN101_Noise_classWise
python unlearn_stage3_test_supervised.py --dataset='cifar10' --pretrained --device='cuda:0' 
### cifar10-poisoned-scratch-ViT_B_16_Noise_sampleWise
python unlearn_stage3_test_supervised.py --dataset='cifar10' --pretrained --device='cuda:0'
### cifar10-poisoned-scratch-ViT_B_16_Noise_classWise
python unlearn_stage3_test_supervised.py --dataset='cifar10' --pretrained --device='cuda:0'  
### cifar10-poisoned-scratch-ViT_B_32_Noise_sampleWise
python unlearn_stage3_test_supervised.py --dataset='cifar10' --pretrained --device='cuda:0' 
### cifar10-poisoned-scratch-ViT_B_32_Noise_classWise
python unlearn_stage3_test_supervised.py --dataset='cifar10' --pretrained --device='cuda:0' 







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


