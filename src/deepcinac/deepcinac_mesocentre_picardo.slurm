#!/bin/sh
#SBATCH -J job_tc
#SBATCH -A b182
#SBATCH -p volta
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding # active l’affinité CPU:GPU
#SBATCH --ntasks-per-node=10
#SBATCH -t 07-00
#SBATCH -o %j.%a.out
#SBATCH -e %j.%a.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=julien.denis3@gmail.com
# chargement des modules
module purge
module load userspace/all
module load python3/3.6.3
module load cuda/10.1
export PYTHONPATH=/scratch/mpicardo/deepcinac/deepcinac_src:$PYTHONPATH
export LD_LIBRARY_PATH=/scratch/mpicardo/deepcinac/libcudnn_10_1/lib64:$LD_LIBRARY_PATH
cd /scratch/mpicardo/deepcinac/src
python3 deep_cinac_training_mesocentre_picardo.py
