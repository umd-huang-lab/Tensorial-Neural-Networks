#!/usr/bin/env bash

function runexp {
    cr=${1}
    mkdir -p outputs/ResNet101/RCP/cr_${cr}/
    python -m torch.distributed.launch --nproc_per_node=4 main_amp.py . --model "ResNet101" --workers 4 --opt-level O1 --batch-size 80 --compression-rate ${cr} --decompose-type "RCP" --epochs 100
}
runexp 0.1
