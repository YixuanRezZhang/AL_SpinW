#!/bin/bash

#SBATCH -J Spinw_simulation
#SBATCH -A p0020164
### SBATCH --mail-type=END
### SBATCH -e /home/<TU-ID>/<project_name>/%x.err.%j
### SBATCH -o /home/<TU-ID>/<project_name>/%x.out.%j
# CPU specification
#SBATCH -n 1                  # 1 process
#SBATCH -c 96                 # 24 CPU cores per process    #  can be referenced as $SLURM_CPUS_PER_TASK​ in the "payload" part
#SBATCH --mem-per-cpu=3000    # Hauptspeicher in MByte pro Rechenkern
#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH -t 24:00:00           # in hours:minutes, or '#SBATCH -t 10' - just minutes

# GPU specification
### SBATCH --gres=gpu
### SBATCH --gres=gpu:v100:2     # 2 GPUs of type NVidia "Volta 100"

# -------------------------------
# your job's "payload" in form of commands to execute, eg.
module purge
## module load gcc cuda

#cd /work/projects/p0020541/Yixuan/ALspinw

source /work/projects/p0020541/Yixuan/spinw/spinw_env/bin/activate
module load matlab/R2023a
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python turbo_exp_swmultipath_AL.py > outinfo

EXITCODE=$?
#  any cleanup and copy commands:
# end this job script with precisely the exit status of your scientific program above:
exit $EXITCODE