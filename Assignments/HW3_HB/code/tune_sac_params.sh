#!/bin/bash
# ----- Sbatch config -----
# Please check https://slurm.schedmd.com/sbatch.html if you have any questions
## Specify the name of the run.
#SBATCH -J tune_sac_params

## Controls the number of replications of the job that are run
## The specific ID of the replication can be accesses with the environment variable $SLURM_ARRAY_TASK_ID
## Can be used for seeding
#SBATCH -a 0

## ALWAYS leave this value to 1. This is only used for MPI, which is not supported now. 
#SBATCH -c 1

## Specify the number of cores and their required memorys. Leave these values at 1.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

## Here you can control the amount of memory that will be allocated for you job. To set this,
## you should run the programm on your local computer first and observe the memory it consumes.
#SBATCH --mem-per-cpu 2000M

## Duration of the job
## Do not allocate more time than you REALLY need. Maximum is 6 hours.
#SBATCH -t 05:00:00

#SBATCH -A kurs00077                                                                                                                                                                                                    
#SBATCH -p kurs00077                                                                                                                                                                                                    
#SBATCH --reservation=kurs00077

## Make sure to create the logs directory, BEFORE launching the jobs !!!
#SBATCH -o /home/tv47cogo/100MHurdles/log_sbatch/SAC_%A_%a.out

# Make sure to activate the conda environment before requesting the job
# conda activate /work/home/kurse/kurs00077/rl_assign3

# You can launch the job using the following:
# sbatch tune_algorithm_params.sh

python tune_algorithm_params.py --config configs/sac.yaml

wait # This will wait until both scripts finish
echo "########################################################################"
echo "...done."