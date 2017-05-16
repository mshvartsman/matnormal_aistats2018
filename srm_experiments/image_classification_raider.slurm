#!/bin/bash
#SBATCH -c 32
#SBATCH -N 1 # All cores on one machine
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ms44@princeton.edu
#SBATCH -t 0-24:00 # (D-HH:MM)
#SBATCH --array=1-24

module switch intel-mpi/intel/5.1.2/64 intel-mpi/intel/2017.0/64
module load python/3.5
source activate tf

python /usr/people/ms44/repos/matnormal_nips2017/srm_experiments/image_classification_raider.py
