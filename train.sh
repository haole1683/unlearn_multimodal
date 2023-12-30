# train generator script
python -m torch.distributed.launch --nproc_per_node=4 --master_port 61201 --use_env train_generator.py --distributed 

python -m torch.distributed.launch --nproc_per_node=4 --master_port 61201 --use_env train_generator_adv.py --distributed 

python -m torch.distributed.launch --nproc_per_node=4 --master_port 61201 --use_env train_generator_text_guide_targeted_at.py --distributed 