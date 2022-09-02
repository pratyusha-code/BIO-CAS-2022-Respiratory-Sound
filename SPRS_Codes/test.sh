#!/bin/bash
#SBATCH -n 12
#SBATCH --mem-per-cpu=2048
#SBATCH --mincpus=35
#SBATCH --gres=gpu:4
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=adithya.edakkadan@students.iiit.ac.in
#SBATCH --mail-type=ALL

module add u18/cuda/10.0
module add u18/cudnn/7.6-cuda-10.0

source /home2/${USER}/.bashrc
conda activate fmcw

mkdir -p /scratch/$USER/
scp ada:/share1/adithyasunil/SPRS/SPRSound_test.zip /scratch/$USER/
unzip /scratch/$USER/SPRSound_test.zip -d /scratch/$USER/

scp ada:/share1/adithyasunil/model_checkpoints_SPRS/event_classification/vgg16_latest/model_event_epoch_95.pt /scratch/$USER/model_event.pt
scp ada://share1/adithyasunil/model_checkpoints_SPRS/record_classification/vgg16_latest/model_event_epoch_95.pt /scratch/$USER/model_record.pt

cd /home2/$USER/resp_sound/codes

python main_test.py 11 /scratch/adithyasunil/SPRSound_test/task2_wav /scratch/adithyasunil/event_out1.json
python main_test.py 12 /scratch/adithyasunil/SPRSound_test/task2_wav /scratch/adithyasunil/event_out2.json
python main_test.py 21 /scratch/adithyasunil/SPRSound_test/task2_wav /scratch/adithyasunil/record_out1.json
python main_test.py 22 /scratch/adithyasunil/SPRSound_test/task2_wav /scratch/adithyasunil/record_out2.json

rsync -av /scratch/adithyasunil/event_out1.json ada:/share1/$USER/SPRS/latest_test
rsync -av /scratch/adithyasunil/event_out2.json ada:/share1/$USER/SPRS/latest_test
rsync -av /scratch/adithyasunil/record_out1.json ada:/share1/$USER/SPRS/latest_test
rsync -av /scratch/adithyasunil/record_out2.json ada:/share1/$USER/SPRS/latest_test
# rsync -av /scratch/adithyasunil/event_out.json ada:/share1/$USER/SPRS/latest_test
