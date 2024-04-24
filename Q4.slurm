#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_002-2024sp-20

## change the partition number to use different number of GPUs
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

##SBATCH --partition=n1s16-v100-2
##SBATCH --gres=gpu:2
##SBATCH --cpus-per-task=16

#SBATCH --partition=n1c24m128-v100-4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24

#SBATCH --time=12:00:00
#SBATCH --output=demo_%j.out
#SBATCH --error=demo_%j.err
#SBATCH --exclusive
#SBATCH --requeue

module purge
module load python/intel/3.8.6
module load anaconda3/2024.02

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/DL23SP/dataset_student.sqsh /tmp
echo "Dataset is copied to /tmp"

conda create -y -n jepa python=3.9 pip  # Use -y to avoid manual confirmation

conda init bash

source activate jepa

pip install torch torchvision torchaudio

pip install .
pip install PyYAML

singularity exec --nv \
--bind /scratch \
--overlay /tmp/dataset_student.sqsh \
/share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
/bin/bash -c "python -m app.main --fname configs/pretrain/vitl16.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3"
