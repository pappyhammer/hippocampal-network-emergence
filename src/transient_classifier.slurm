#!/bin/sh
#SBATCH -J job_tc
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding # active l’affinité CPU:GPU
#SBATCH --ntasks-per-node=1
#SBATCH -t 07-00
#SBATCH -o %j.%a.out
#SBATCH -e %j.%a.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=julien.denis3@gmail.com
# chargement des modules
module purge
module load userspace/all
module load python3/3.6.3
module load cuda/9.0
export PYTHONPATH=/scratch/mpicardo/packages_homemade:$PYTHONPATH
export LD_LIBRARY_PATH=/scratch/mpicardo/hne_project/libcudnn/usr/lib64:$LD_LIBRARY_PATH
cd /scratch/mpicardo/hne_project/code
python3 transient_classifier_main.py
