### Train generator
# distribute
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 --master_port 61201 --use_env train_generator.py --distributed 
python -m torch.distributed.launch --nproc_per_node=4 --master_port 61201 --use_env train_generator.py --distributed --checkpoint /remote-home/songtianwei/research/unlearn_multimodal/output/train_generator_max_loss/checkpoint_epoch_10.pth 

### Generate poison dataset
python -m torch.distributed.launch --nproc_per_node=4 --master_port 61201 --use_env poison_data_gen.py --distributed --checkpoint /remote-home/songtianwei/research/unlearn_multimodal/output/train_generator_linf_max_loss/checkpoint_epoch_1.pth
python -m torch.distributed.launch --nproc_per_node=4 --master_port 61201 --use_env poison_data_gen.py --debug --distributed

### Finetune CLIP in downstream poison dataset
python -m torch.distributed.launch --nproc_per_node=4 --master_port 61201 --use_env retrieval_by_CLIP.py --distributed --freeze_encoder text
python -m torch.distributed.launch --nproc_per_node=4 --master_port 61201 --use_env retrieval_by_CLIP.py --distributed --poisoned --freeze_encoder text


## 4.8.1 transformers 之前的