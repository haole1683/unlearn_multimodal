# python unlearn_test_finetune_clip.py --device cuda:0 --finetune_dataset myLaion
# python unlearn_test_finetune_clip.py --device cuda:1 --poisoned --finetune_dataset myLaion
# python unlearn_test_finetune_clip.py --device cuda:2 --finetune_dataset cifar10
# python unlearn_test_finetune_clip.py --device cuda:3 --poisoned --finetune_dataset cifar10

# 2024-4-13
python unlearn_stage1_train_all_g.py --clip_model=both --device=cuda:0 --overwrite

# stage1 - train generator script 
## For RN50*4
accelerate launch --config_file=accelerate_config.yaml unlearn_stage1_train_all_g_dis.py --clip_model=RN50x4 --trainset=all --batch_size=16 --overwrite
## For Other model version
accelerate launch --config_file=accelerate_config.yaml unlearn_stage1_train_all_g_dis.py --clip_model=both --trainset=all --batch_size=32 --overwrite 

# stage2 - generate noise from generator
python unlearn_test_generator_gen_noise.py --clip_model=RN50x4 --device=cuda:0 --dataset=stl10 --gen_which=both
python unlearn_test_generator_gen_noise.py --clip_model=RN50x4 --device=cuda:0 --dataset=cifar10
python unlearn_test_generator_gen_noise.py --device="cuda:0" --gen_which=all --overwrite --update_z_freq=1000000 --text_prompt_stragegy=fixed

# stage3 - test noise script
## 报错。
python unlearn_test_finetune_clip.py --device="cuda:0" --clip_model=RN50x4 --batch_size=16
python unlearn_test_finetune_clip.py --device="cuda:1" --clip_model=RN50x4 --poisoned --noise_path="/remote-home/songtianwei/research/unlearn_multimodal/output/unlearn_stage2_generate_noise/both/noise_gen2_46221-3-224-224_all_both.pt"
