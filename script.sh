# python unlearn_test_finetune_clip.py --device cuda:0 --finetune_dataset myLaion
# python unlearn_test_finetune_clip.py --device cuda:1 --poisoned --finetune_dataset myLaion
# python unlearn_test_finetune_clip.py --device cuda:2 --finetune_dataset cifar10
# python unlearn_test_finetune_clip.py --device cuda:3 --poisoned --finetune_dataset cifar10

# 2024-4-13
python unlearn_stage1_train_all_g.py --clip_model=both --device=cuda:0 --overwrite

# cur sh script
accelerate launch --config_file=accelerate_config.yaml unlearn_stage1_train_all_g_dis.py --clip_model=both --overwrite