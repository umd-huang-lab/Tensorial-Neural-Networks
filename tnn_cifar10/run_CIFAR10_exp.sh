#!/usr/bin/env bash



function runexp {

dataset=${1}
decompose_type=${2}
cr=${3}
batch_size=${4}




#--resume ${decompose_type}/cr${cr}/model_best.pth.tar

expname=/nfshomes/xliu1231/tnn_cifar10/logs/CIFAR10_${decompose_type}_cr${cr}_FULL


python -m torch.distributed.launch --nproc_per_node=4 CIFAR10_exp.py  --workers 4 --opt-level O1 --batch-size ${batch_size} --compression-rate ${cr} --decompose-type ${decompose_type} --epoch-num 100  --print_freq 10 --learning-rate 0.05 #> ${expname}.log 2>&1

}
#         dataset   decompose type  compression rate   batch_size
runexp "CIFAR10" "CP" 0.01 256
