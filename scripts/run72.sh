#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=24:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=test_ppo_transx10_vac_60
#SBATCH --output=test_ppo_transx10_vac_60.out

source ../venvs/epipolicy/bin/activate

module load python/intel/3.8.6
module load openmpi/intel/4.0.5
time python3 ppo_kernel.py  --exp-name test_ppo_transx10_vac_60 --gym-id new_jsons_transx10/SIRV_A --vac-starts 60