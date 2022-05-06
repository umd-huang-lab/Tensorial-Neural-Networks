#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=tnnMulti                                     # sets the job name if not set from environment
#SBATCH --array=0-4
#SBATCH --output slurm_logs/%x_%A_%a.out                                  # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error slurm_logs/%x_%A_%a.err                                    # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --time=36:00:00                                         # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=furongh                                  # set QOS, this will determine what resources can be requested
#SBATCH --qos=high                                         # set QOS, this will determine what resources can be requested
#SBATCH --partition=dpart
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem 128gb                                              # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-user=xliu1231@umd.edu
#SBATCH --mail-type=END,FAIL               # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE,

function runexp {

dataset="CIFAR10"
decompose_type=${1}
cr=${2}
batch_size=${3}
lr=${4}




#expname=/nfshomes/xliu1231/tnn_cifar10/logs/${dataset}_${decompose_type}_cr${cr}_${batch_size}_${lr}
expname=/nfshomes/xliu1231/tnn_cifar10/logs/${dataset}_${decompose_type}_cr${cr}_Pytorch_${path}


python -m torch.distributed.launch --nproc_per_node=4 CIFAR10_exp.py  --workers 4 --opt-level O1 --batch-size ${batch_size} --compression-rate ${cr} --decompose-type ${decompose_type} --epoch-num 100  --print_freq 10 --learning-rate ${lr} > ${expname}.log 2>&1



}



source /cmlscratch/xliu1231/anaconda3/etc/profile.d/conda.sh
conda activate base

#types=("RCP" "RCP_LTR")
#lrs=(0.1 0.05 0.3)
crs=(0.05 0.1 0.2 0.5 1.0)
#bs=(4096 8192 15000 30000 60000)

idx=${SLURM_ARRAY_TASK_ID}
#tidx=$(( (${SLURM_ARRAY_TASK_ID}) % 2 ))
#cidx=$(( (${SLURM_ARRAY_TASK_ID}) / 2 % 6 ))





runexp "None" ${crs[$idx]} 256 0.05 ${path[$idx]}
