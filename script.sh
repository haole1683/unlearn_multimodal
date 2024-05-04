# python unlearn_test_finetune_clip.py --device cuda:0 --finetune_dataset myLaion
# python unlearn_test_finetune_clip.py --device cuda:1 --poisoned --finetune_dataset myLaion
# python unlearn_test_finetune_clip.py --device cuda:2 --finetune_dataset cifar10
# python unlearn_test_finetune_clip.py --device cuda:3 --poisoned --finetune_dataset cifar10

# 2024-4-13
python unlearn_stage1_train_all_g.py --clip_model=both --device=cuda:0 --overwrite



# stage1 - train generator script 
## For RN50*4, the batch size need half(25G memory)
accelerate launch --config_file=accelerate_config.yaml unlearn_stage1_train_all_g_dis.py --clip_model=RN50x4 --trainset=all --batch_size=16 --overwrite
## For Other model version
accelerate launch --config_file=accelerate_config.yaml unlearn_stage1_train_all_g_dis.py --clip_model=RN50 --trainset=all --batch_size=8 --overwrite 
## For both model version(RN101 + ViT-B/16)
accelerate launch --config_file=accelerate_config.yaml unlearn_stage1_train_all_g_dis.py --clip_model=both --trainset=all --batch_size=32 --overwrite 


# stage2 - generate noise from generator
# Note: update_z_fre set max, not update z
# text_prompt set fixed , use fixed text prompt
# gen1 : generate noise for supervised model
# gen2 : generate noise for myLaion dataset
python unlearn_stage2_generator_gen_noise.py --device='cuda:0' --clip_model=ViT-B/16 --gen_which=all --overwrite --update_z_freq=1000000 --text_prompt_stragegy=fixed
python unlearn_stage2_generator_gen_noise.py --device='cuda:2' --clip_model=RN50 --gen_which=all --overwrite --update_z_freq=1000000 --text_prompt_stragegy=fixed


# stage3 - test noise script
## 报错。
python unlearn_stage3_test_finetune_clip.py --device="cuda:0" --clip_model=RN50x4 --batch_size=16
python unlearn_stage3_test_finetune_clip.py --device="cuda:1" --clip_model=RN50x4 --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/both/noise_gen2_46221-3-224-224_all_both.pt"
## self-supervised model
python unlearn_stage3_test_self_supervised.py --dataset='cifar10' --device='cuda:0'
python unlearn_stage3_test_self_supervised.py --dataset='cifar10' --device='cuda:1' --poisoned --noise_path="./output/unlearn_stage2_generate_noise/ViT-B-16/cifar10/noise_gen1_ViT-B-16_cifar10_all.pt"
python unlearn_stage3_test_self_supervised.py --dataset='stl10' --device='cuda:2'
python unlearn_stage3_test_self_supervised.py --dataset='stl10' --device='cuda:3' --poisoned --noise_path="./output/unlearn_stage2_generate_noise/ViT-B-16/stl10/noise_gen1_ViT-B-16_stl10_all.pt"