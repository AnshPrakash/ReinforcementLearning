#!/bin/bash

#SBATCH -J minatar_dqn_variants
## Specify the name of the run.
#SBATCH -a 0
## Controls the number of replications of the job that are run. E.g. use #SBATCH -a 0-2 for 3 replications
## The specific ID of the replication can be accesses with the environment variable $SLURM_ARRAY_TASK_ID
## Can be used for seeding
#SBATCH -n 1  
## ALWAYS leave this value to 1. This is only used for MPI, which is not supported now. 
#SBATCH -c 1
## Specify the number of cores per job.
#SBATCH --mem-per-cpu 2000
## Here you can control the amount of memory that will be allocated for your job. To set this,
## you should run the programm on your local computer first and observe the memory it consumes.
#SBATCH -t 02:00:00
## Do not allocate more time than you REALLY need. Maximum is 6 hours.

#SBATCH -A kurs00077                                                                                                                                                                                                    
#SBATCH -p kurs00077                                                                                                                                                                                                    
#SBATCH --reservation=kurs00077
## Comment these lines out if running locally!

#SBATCH -o [YOUR_LOCATION]/logs_sbatch/%A_%a.out
#SBATCH -e [YOUR_LOCATION]/logs_sbatch/%A_%a.err
## Make sure to create the logs directory (log_sbatch) in [YOUR_LOCATION], BEFORE launching the jobs. [YOUR_LOCATION] needs to be specified by you.

ALG=$1
EXP_NAME=$2

python minatar_dqn_variants.py --n_exp 5 --agent_class ${ALG} --exp_name ${EXP_NAME}