#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=24:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=test_sac_transx100_vac_0
#SBATCH --output=test_sac_transx100_vac_0.out

source ../venvs/epipolicy/bin/activate

module load python/intel/3.8.6
module load openmpi/intel/4.0.5
time python3 sac_kernel.py  --exp-name test_sac_transx100_vac_0 --gym-id new_jsons_transx100/SIRV_A --vac-starts 0