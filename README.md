
# Convolutional Tensorial Neural Networks

We implement convolutional tensorial neural networks (TNNs), a generalization of existing neural networks by extending tensor operations on low order operands to those on high order operands.  Training through this module will automatically evaluate a network as a tensor graph (with specified decompose_type and compression-rate). 

The optimal sequencer which lives in torch_conv_einsumfunc.py is automatically deployed to evaluate the tensorial forward and backward pass sequences in a FLOPs-optimal manner. Note that the optimal sequencer forks and modifies components of opt-einsum (https://github.com/dgasmith/opt_einsum) to provide support for convolutions. 

opt-einsum: 
Daniel G. A. Smith and Johnnie Gray, opt_einsum - A Python package for optimizing contraction order for einsum-like expressions. Journal of Open Source Software, 2018, 3(26), 753
DOI: https://doi.org/10.21105/joss.00753


# Image Classification


To run a standalone ResNet CIFAR-10 experiment, relevant files live in tnn_cifar10. 

Run the following and modify inputs as necessary (which are documented in CIFAR10_exp.py): 

```python=

python -m torch.distributed.launch --nproc_per_node=4 CIFAR10_exp.py  --workers 4 --opt-level O1 --batch-size ${batch_size} --compression-rate ${cr} --decompose-type ${decompose_type} --epoch-num 100  --print_freq 10 --learning-rate ${lr} > ${expname}.log 2>&1

```
An optional --model argument is available, default is ResNet34. 
model: ResNetX
X=18,34,50,101,152

A wide variety of tensor decomposition types are available:
decompose_type: RCP CP RTK TK RTT TT RTR TR
CP: Canonical Polyadic  (Kolda et al., 2009)
RCP: Reshaped Canonical Polyadic (Su et al., 2018)
TK: Tucker (Kolda et al., 2009)
RTK: Reshaped Tucker (Su el al., 2018)
TT: Tensor Train (Oseledets, 2011)
RTT: Reshaped Tensor Train (Su et al., 2018)
TR: Tensor Ring (Zhao et al., 2016)
RTR:  Reshaped Tensor Ring (Su et al., 2018)



# Video classification

TNN Two-Stream Video Classification:
Relevant files live in tnn_VC

1. Pretrain TNN-ResNet101 on ImageNet:
    a. Create links to ImageNet ILSVRC2012 training and validation sets at ./ResNet/train and ./ResNet/val
    b. Run bash ./ResNet/run.sh
2. Train TNN-two-stream network:
    a. Create link to preprocessed UCF101 dataset at ./UCF101
    b. Run bash ./run.sh

Currently, these experiments have only been tested and certified on ResNet101 with decompose_type RCP. While it is possible to change the model and decompose_type arguments, the behavior might not be as intended. 

# conv_einsum


To run a standlone conv_einsum (i.e., evaluate an einsum string containing convolutions), import opt-einsum and run something like the following:

```
A=np.random.rand(4,7,9)
B=np.random.rand(10,5)
C=np.random.rand(5,4,2)
D=np.random.rand(6,8,9,2)
path_info = conv_einsum.contract_path("ijk,jl,lmq, njpq->ijknp|j", A, B, C, D)
print(path_info[1])
```
For further examples, especially relating to FLOPs calculations within ResNet-34, see the conv_playground iPython notebook.  
