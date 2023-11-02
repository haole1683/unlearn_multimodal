### Train generator
# distribute
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 --master_port 61201 --use_env train_generator.py --distributed 

### Generate poison dataset
python -m torch.distributed.launch --nproc_per_node=4 --master_port 61201 --use_env poison_data_gen.py --distributed

### Finetune CLIP in downstream poison dataset
python -m torch.distributed.launch --nproc_per_node=4 --master_port 61201 --use_env retrieval_by_CLIP.py --distributed --freeze_encoder text