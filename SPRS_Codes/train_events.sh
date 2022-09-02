#!/bin/bash
#SBATCH -n 35
#SBATCH --mem-per-cpu=2048
#SBATCH --mincpus=30
#SBATCH --gres=gpu:4
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=adithya.edakkadan@students.iiit.ac.in
#SBATCH --mail-type=ALL

module add u18/cuda/10.0
module add u18/cudnn/7.6-cuda-10.0

source /home2/${USER}/.bashrc
conda activate fmcw

mkdir -p /scratch/$USER/
scp ada:/share1/$USER/SPRS/event.zip /scratch/$USER/
unzip /scratch/$USER/event.zip -d /scratch/$USER/

cd /home2/$USER/resp_sound/codes

mkdir -p /scratch/$USER/model_checkpoints_SPRS/event_classification/vgg16

python main_train_events.py

rsync -av /scratch/adithyasunil/model_checkpoints_SPRS/ ada:/share1/$USER/model_checkpoints_SPRS
