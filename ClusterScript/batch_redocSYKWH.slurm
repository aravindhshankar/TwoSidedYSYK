#!/bin/bash
#SBATCH --job-name=redo_cSYK_WH
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="shankar@lorentz.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=30G
#SBATCH --time=2-23:00:00
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
echo "[$SHELL] Running Python script"
#
python cSYKWH.py 
#
echo "[$SHELL]#### SLURM script finished. "
