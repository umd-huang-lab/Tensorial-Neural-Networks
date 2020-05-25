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
    CONVOLUTION
};

enum class ConvolutionType {
    
   
    // agrees with numpy's convolution same option
    // takes the middle M values of the full convolution, with possibly 1 extra on the left,
    // where M is the maximum length of the vectors.
    // The full convolution is obtained by taking as entries the coefficients of the polynomial
    // product of the two vectors
    // so the full convolution is defined F_i = sum_{i1} A_{i1}B_{i-i1}, and the same
    // convolution is defined S_i = F_{i + (N-1)/2}, where N is the minimum length of the
    // two vectors, and i ranges over M
    SAME,  

    // \todo
    // CYCLIC should be considered broken because I have not seen a standard
    // convention for treating the cyclic convolution of differently sized vectors, which
    // is both commutative and at least effectively associative (aside from the boundary)
    CYCLIC 
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


    void SetConvolutionType(ConvolutionType type);

    void AddNode(std::string name, size_t order);

    
    
    struct NodeMode {
        std::string node_name;
        size_t mode;
    };

    /**
     * If output_mode < 0 then this is not an output edge
     */
    void AddEdge(std::string edge_name, 
                 std::vector<NodeMode> edge_parts,
                 Operation operation = Operation::CONTRACTION,
                 int output_mode = -1);
                              
     
 

    // may want to store the index of the node here for optimization purposes
    struct EdgePart {
        std::string node_name; 
        size_t mode;
        Operation operation = Operation::CONTRACTION;
    };

    /**
     * If output_mode < 0 then this is not an output edge 
     */
    void AddMultiOperationEdge(std::string edge_name, 
                               std::vector<EdgePart> edge_parts,
                               int output_mode = -1);
                               
   

    //void SetOutputEdge(std::string edge_name, size_t output_mode);
    // actually, I think they should specify if it's an output edge when they add the edge
    
    

    Tensor Evaluate(const std::vector<Tensor>& tensors);


    size_t NumNodes();
    
    bool HasOnlyContraction();
    bool HasNoMultiOperationEdges();

    private:
    ConvolutionType convolution_type = ConvolutionType::SAME;

    std::vector<size_t> nodes; // nodes<order of node> 

    struct NodeIndexMode {
        size_t index;
        size_t mode;
    };

    struct InternalEdgePart {
        //std::string node_name; 
        size_t node_index;
        size_t mode;
        Operation operation = Operation::CONTRACTION;

    };
 
    struct Edge {
        // \todo might want InternalEdgePart to store the index of the node
        std::vector<InternalEdgePart> edge_parts;
        int output_mode = -1; //If output_mode < 0 then it's not an output edge
            // \todo store output_mode here?

        /**
         * If there is a single Operation type, and single_operation_out is not nullptr,
         * then single_operation_out is set to the single Operation type 
         */
        bool HasSingleOperation(Operation* single_operation_out = nullptr);

        bool HasOnlyContraction();

        bool HasOnlyConvolution();

        // probably want to store this instead of computing it in an inner loop
        size_t NumConvolutionParts();
        
        size_t GetModeSize(const std::vector<Tensor>& tensors, size_t edge_part = 0);
        void GetMinMaxModeSizes(const std::vector<Tensor>& tensors, size_t& min, size_t& max);

        /**
         * Calculates the offset which indexes the first of the the middle components 
         * from the full convolution. See Numpy's "same" option for convolution.
         */
        size_t GetSAMEOffset(const std::vector<Tensor>& tensors);

    };  

    // \todo it's perhaps better to separate the output edges and the nonoutput edges
    // it could be best to separate all of the types of elements
    //like std::vector<Edge> output_edges;
    //     std::vector<Edge> nonoutput_edges;
    // if you do this then you need to have two maps edge_name_index, and probably
    // double the number of variables in other places
    // an alternative is to insert the edges so that the output edges are at the beginning
    // and the nonoutput edges are at the end. Then you have to remap edge_name_index on each
    // insert

    std::vector<Edge> edges;

    /*
     * the (node, mode) pair at the first index becomes the first mode in the output,
     * the second (node, mode) pair becomes the second mode of the output, etc...
     * defaults to the same order they're added to the network
     * \todo you should be able to name the output modes... 
     * maybe you don't set the order of the node when you add the
     * node, but when you add edges with only one node...
     *
     * \todo this is redundant with edges but perhaps more accessible
     */
    std::vector<NodeIndexMode> nonoutput_modes_map;

    struct OutputMode {
        size_t index; // index into the node array or edge array 
        size_t output_mode; // if I store them in order this is redundant \todo
        bool is_edge = true; // the output mode comes from an edge
        size_t node_mode = -1; // n/a if is_edge == true
    };

    std::vector<OutputMode> output_modes; // \todo remove nonoutput_modes_map
    std::vector<size_t> CalcOutputTensorSize(const std::vector<Tensor>& tensors);
    

    // fixed length strings are a possible optimization 
    std::map<std::string, size_t> node_name_index; 
    std::map<std::string, size_t> edge_name_index; 

    //std::vector<NodeIndexMode> CalcNodeOutputModesMap(); 
    

    // \todo I'd like to rename this so it implies what it's trying to do, which is 
    // to evalute the sum of products by definition
    Tensor EvaluateEinsumNaive(const std::vector<Tensor>& tensors);

    // this one is for networks which include convolution, I'm separating them because
    // I assume the naive evaluation method may be different if convolution is involved,
    // but I may later remove the einsum only naive function if it's a pure subset of this
    Tensor EvaluateConvNaive(const std::vector<Tensor>& tensors);
 
};


} // OPS

#endif // TENSOR_NETWORK_H
