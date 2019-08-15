#!/bin/bash
#SBATCH --job-name=GCN_L_20190604
#SBATCH --output=/exports/lkeb-hpc/zzhai/project/work/AV_dl/arteryveinseparation/2_CNN_L_GCN/slurm_output/GCN_L_20190601.txt
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --partition=gpu
#SBATCH --nodelist=res-hpc-gpu01
#SBATCH --mail-type=ALL
#SBATCH --mail-user z.zhai@lumc.nl
#SBATCH --gres=gpu:1
#SBATCH --time=0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/exports/lkeb-hpc/zzhai/program/cudnn7.4-for-cuda9.0/cuda/lib64/
source /exports/lkeb-hpc/zzhai/python/p3tf/bin/activate

echo "on Hostname = $(hostname)"
echo "on GPU      = $CUDA_VISIBLE_DEVICES"
echo
echo "@ $(date)"
echo
python /exports/lkeb-hpc/zzhai/project/work/AV_dl/arteryveinseparation/2_CNN_L_GCN/Training_gcn3D_AV_keras_generator.py --where_to_run Cluster