#!/bin/bash

#SBATCH --j=bye-vae
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=15G
#SBATCH -w augi1
#SBATCH -p batch
#SBATCH -t 240:00:00
#SBATCH -o logs/%N_%x_%j.out
#SBTACH -e %x_%j.err

source /data/kayoung/init.sh
conda activate hi-vae

python /data/kayoung/bye-vae/main.py