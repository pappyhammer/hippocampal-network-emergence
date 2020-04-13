#!/bin/sh
#SBATCH -J job_tada
#SBATCH -A b182
#SBATCH -p kepler
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding # active l’affinité CPU:GPU
#SBATCH --ntasks-per-node=4
#SBATCH -t 07-00
#SBATCH -o %j.%a.out
#SBATCH -e %j.%a.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=julien.denis3@gmail.com
# chargement des modules
# volta or kepler
module purge
module load userspace/all
module load python3/3.6.3
module load cuda/9.0
#module load gcc/7.3.0
export PYTHONPATH=/scratch/edenis/deeptada/deeptada_src:$PYTHONPATH
#export LD_LIBRARY_PATH=/scratch/edenis/deepcinac/libcudnn_10_1/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/scratch/edenis/deepcinac/libcudnn_9_0/:$LD_LIBRARY_PATH
cd /scratch/edenis/deeptada/src
python3 tada_training_mesocentre.py