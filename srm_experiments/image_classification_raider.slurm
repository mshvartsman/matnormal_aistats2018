#!/bin/bash
#SBATCH -c 32
#SBATCH -N 1 # All cores on one machine
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ms44@princeton.edu
#SBATCH -t 0-24:00 # (D-HH:MM)
#SBATCH --array=1-27

python3 /usr/people/ms44/repos/matnormal_nips2017/srm_experiments/image_classification_raider.py
