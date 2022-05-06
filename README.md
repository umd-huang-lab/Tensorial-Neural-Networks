# Tensorial-Neural-Networks
We implement tensorial neural networks (TNNs), a generalization of existing neural networks by extending tensor operations on low order operands to those on high order operands. 


To run the code, change the corresponding content in 

```python=

sbatch run_multi_exp.sh

```

or 

```python=

python -m torch.distributed.launch --nproc_per_node=4 CIFAR10_exp.py  --workers 4 --opt-level O1 --batch-size ${batch_size} --compression-rate ${cr} --decompose-type ${decompose_type} --epoch-num 100  --print_freq 10 --learning-rate ${lr} > ${expname}.log 2>&1


```

decompose_type: RCP CP RTK TK RTT TT RTR TR
