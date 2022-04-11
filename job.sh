#!/bin/sh
#
# Matlab submit script for Slurm.
#
#
#SBATCH -A apam               # Replace ACCOUNT with your group account name 
#SBATCH -J NLSW          # The job name
#SBATCH -c 1                     # Number of cores to use (max 32)
#SBATCH -t 1-23:30                # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=5G         # The memory the job will use per cpu core

module load matlab

touch RUNNING

echo "Launching a Matlab run"
date

#define parameters
q=1
N=70
T=1
alpha=0.5
tau=0.5
beta=0.5

#Command to execute Matlab code
matlab -nosplash -nodisplay -nodesktop -r "NLSWfluxmanu_ssprk54($q,$N,$T,$alpha,$tau,$beta)"

touch done

\rm RUNNING

# End of script

