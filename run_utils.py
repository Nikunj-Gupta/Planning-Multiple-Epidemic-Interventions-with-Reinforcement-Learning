import os, yaml, argparse
from pathlib import Path 
from itertools import count 

dumpdir = "scripts/" 
if not os.path.isdir(dumpdir):
    os.mkdir(dumpdir)
fixed_text = "#!/bin/bash\n"\
             "#SBATCH --nodes=1\n"\
             "#SBATCH --cpus-per-task=16 \n"\
             "#SBATCH --time=24:00:00\n"\
             "#SBATCH --mem=20GB\n"\
            #  "#SBATCH --gres=gpu:1\n"

for algo in ['sac', 'ppo']: 
    for initial_condition in ['jsons', 'new_jsons_transx10', 'new_jsons_transx100', 'new_jsons_initial_infectedx10']: 
        for scenario in ['SIRV_A', 'SIRV_B', 'SIR_A', 'SIR_B', 'COVID_A', 'COVID_B', 'COVID_C']: 
            for vac_starts in [0, 60]: 
                exp = '_'.join(['test', algo, initial_condition.split('_')[-1], 'vac', str(vac_starts)])
                command = fixed_text + "#SBATCH --job-name="+exp+"\n#SBATCH --output="+exp+".out\n"
                command += "\nsource ../venvs/epipolicy/bin/activate\n"\
                    "\nmodule load python/intel/3.8.6\n"\
                    "module load openmpi/intel/4.0.5\n"\
                    "time python3 " 
                command += 'sac_kernel.py ' if algo=='sac' else 'ppo_kernel.py '
                command = ' '.join([
                    command, 
                    '--exp-name', exp, 
                    '--gym-id', initial_condition+'/'+scenario, 
                    '--vac-starts ' + str(vac_starts)
                ]) 
                # print(command) 
                log_dir = Path(dumpdir)
                for i in count(1):
                    temp = log_dir/('run{}.sh'.format(i)) 
                    if temp.exists():
                        pass
                    else:
                        with open(temp, "w") as f:
                            f.write(command) 
                        log_dir = temp
                        break 