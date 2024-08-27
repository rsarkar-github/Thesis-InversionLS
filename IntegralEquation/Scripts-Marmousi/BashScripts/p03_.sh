#!/usr/bin/bash
#SBATCH --job-name=p02_
#SBATCH --output=p02_.%j.out
#SBATCH --error=p02_.%j.err
#SBATCH --time=24:00:00
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH -c 4

echo "hostname: " $HOSTNAME

source ~/.bashrc
conda activate py39
cd /home/research/rsarkar/Research

python -m InversionLS.IntegralEquation.Scripts-Marmousi.p03_calculate_green_func