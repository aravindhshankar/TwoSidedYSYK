#!/bin/bash
#SBATCH --job-name=MajWH
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="shankar@lorentz.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --partition=cpu-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#
module load Python/3.10.8-GCCcore-12.2.0
#
source $HOME/WH_env/bin/activate
#
echo "[$SHELL] ## This is $SLURM_JOB_USER on #HOSTNAME and job ID $SLURM_JOB_ID"
export CWD=$(pwd)
echo "[$SHELL] Current working directory: $CWD"
#
export file_name="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
echo "[$SHELL] Running Python script"
#
python ShOut_ClusterMajWH.py "$file_name"
#
echo "[$SHELL]#### SLURM script finished. "



