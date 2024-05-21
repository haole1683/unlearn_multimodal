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
python unlearn_stage3_test_supervised.py --dataset='cifar10' --device='cuda:0' 
python unlearn_stage3_test_supervised.py --dataset='cifar10' --device='cuda:1' --poisoned
python unlearn_stage3_test_supervised.py --dataset='stl10' --device='cuda:2'
python unlearn_stage3_test_supervised.py --dataset='stl10' --device='cuda:3' --poisoned








## 测试一下到底CLIP能不能from_scratch训练。。。调整一下学习率？？
python unlearn_stage3_test_finetune_clip.py --device="cuda:3" --clip_model=RN50 --batch_size=256 --output_dir="./output/unlearn_temp_test2/" --from_scratch --lr="1e-3"