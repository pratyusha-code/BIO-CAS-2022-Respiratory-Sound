#!/bin/bash
#SBATCH -n 12
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
scp ada:/share1/$USER/SPRS/SPRSound.zip /scratch/$USER/
unzip /scratch/$USER/SPRSound.zip -d /scratch/$USER/

cd /home2/$USER/resp_sound/codes

python main_train_data.py

cd /scratch/$USER/
zip -r event.zip event/
zip -r record.zip record/

rsync -av /scratch/$USER/event.zip ada:/share1/$USER/SPRS/dev/
rsync -av /scratch/$USER/record.zip ada:/share1/$USER/SPRS/dev/
