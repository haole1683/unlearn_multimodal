python unlearn_test_finetune_clip.py --device cuda:0 --finetune_dataset myLaion
python unlearn_test_finetune_clip.py --device cuda:1 --poisoned --finetune_dataset myLaion
python unlearn_test_finetune_clip.py --device cuda:2 --finetune_dataset cifar10
python unlearn_test_finetune_clip.py --device cuda:3 --poisoned --finetune_dataset cifar10