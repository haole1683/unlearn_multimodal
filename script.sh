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

# train data on clean dataset: 
### Flickr-PASCAL dataset ###
python -m torch.distributed.launch --nproc_per_node=1 --master_port 61201 --use_env retrieval_by_CLIP.py --distributed --config configs/clip_poison_pascal.yaml --overload_config --output_dir output/pascal_sheep2aeroplane_1.0/ --poisoned_file poisoned_data/pascal_train_sheep2aeroplane_1.0.json --target_txt_cls sheep --target_img_cls aeroplane --poisoned_goal sheep2aeroplane 

## 4.8.1 transformers 之前的



## Pretrain on the self-cifar-10 dataset
python -m torch.distributed.launch --nproc_per_node=4 --master_port 61201 --use_env pretrain.py --distributed 
