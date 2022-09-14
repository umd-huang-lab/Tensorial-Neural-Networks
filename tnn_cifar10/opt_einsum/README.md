# conv_einsum path evaluation
This modifies the package opt-einsum to handle convolutions. 

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
