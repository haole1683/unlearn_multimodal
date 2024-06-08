#!/bin/bash
#SBATCH --account=punim0619
#SBATCH --job-name=unlearn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 #or more, up to 4.
#SBATCH --time=1-23:20:00
#SBATCH --partition=gpu-a100 
#SBATCH --mem=64G

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi


# Run the job from this directory:
cd /home/xingjun/Python/unlearn_multimodal/unlearn_multimodal

# The modules to load:
conda activate pytorch2-unlearn
nvidia-smi

python unlearn_stage3_test_finetune_clip_coco.py --device="cuda:0" --clip_model=RN50 --batch_size=1024 --output_dir="./output/unlearn_train_from_scratch_on_coco/" --from_scratch --lr="1e-3" --finetune_dataset=coco