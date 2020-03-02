#ifndef TENSOR_NETWORK_H
#define TENSOR_NETWORK_H

#include <map>
#include <vector>
#include <string>




/*
Thoughts:
- With einsum, the tensor size information is not contained in the einsum string, but only the
  tensors. Should a TensorNetwork keep track of the tensor sizes (making it more type safe,
  but less generic), or try to emulate einsum? Right now I'll go with emulating einsum

- We could support generic operations on the cpu side (though I don't know how it would work
  with gpu code) at the cost, I think, of an extra indirection 

- TensorNetworkDefinition vs TensorNetwork?
  - I went with Definition to emphasize that it's relatively independent of tensor sizes...
    but maybe it's more intuitive to call it TensorNetwork
*/



namespace OPS {

class Tensor;

enum class Operation {
    CONTRACTION,
    CONVOLUTION // multiple convolution types? or perhaps this needs to be generic
};

 

/**
 * TensorNetworkDefinition encapsulates the definition of a convolutional tensor network.
 * It does not manage the memory of actual tensors.
 */
class TensorNetworkDefinition {
    
    public:
    TensorNetworkDefinition() = default;
    TensorNetworkDefinition(std::string einsum_definition, 
                            std::map<Operation, std::string> operations);

    void AddNode(std::string name, size_t order); 
 
    void AddEdge(std::string name,
                 std::string node1_name, size_t node1_mode, 
                 std::string node2_name, size_t node2_mode,
                 Operation operation = Operation::CONTRACTION);



    


    
    Tensor Evaluate(const std::vector<Tensor>& tensors);
    

    size_t NumNodes();
    size_t OutputOrder(); // \todo this might sound like a command instead of a getter
    bool HasOnlyContraction();

    private:
    std::vector<size_t> nodes; // nodes<order of node>
    struct TensorNetworkEdge {
        size_t node1_index; 
        size_t node1_mode; 
        size_t node2_index; 
        size_t node2_mode; 
        Operation operation = Operation::CONTRACTION;
    };
    // \todo it may be easier to have an edge for every mode in the network, even if
    // it's an output node (so is connected to only one mode) 
    std::vector<TensorNetworkEdge> edges;
    

    // the (node, mode) pair at the first index becomes the first mode in the output,
    // the second (node, mode) pair becomes the second mode of the output, etc...
    // defaults to the same order they're added to the network
    // \todo you should be able to name the output modes... maybe you don't set the order of the node when you add the
    // node, but when you add edges with only one node...
    
    // \todo this is redundant with edges
    std::vector<std::vector<size_t>> nonoutput_modes_map;
    
    

    // fixed length strings are a possible optimization 
    std::map<std::string, size_t> node_name_index; 
    std::map<std::string, size_t> edge_name_index; 

    std::vector<std::vector<size_t>> CalcOutputModesMap();
    std::vector<std::vector<size_t>> CalcSortedNonOutputModesMap();
    Tensor EvaluateEinsumNaive(const std::vector<Tensor>& tensors);


    
};


Tensor Evaluate(std::string einsum_definition, 
                std::map<Operation, std::string> operations,
                const std::vector<Tensor>& tensors);
    


} // OPS

#endif // TENSOR_NETWORK_H
