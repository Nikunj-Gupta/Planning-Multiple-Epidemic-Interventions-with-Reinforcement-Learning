#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=24:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=test_ppo_jsons_vac_0
#SBATCH --output=test_ppo_jsons_vac_0.out

source ../venvs/epipolicy/bin/activate

module load python/intel/3.8.6
module load openmpi/intel/4.0.5
time python3 ppo_kernel.py  --exp-name test_ppo_jsons_vac_0 --gym-id jsons/COVID_A --vac-starts 0