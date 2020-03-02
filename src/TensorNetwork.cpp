#include <iostream>

#include "TensorNetwork.h"
#include "Tensor.h"
#include "HelpersTypedefs.h"

namespace OPS {



TensorNetworkDefinition::TensorNetworkDefinition(std::string einsum_definition, 
                                                 std::map<Operation, std::string> operations) 
{

}



void TensorNetworkDefinition::AddNode(std::string name, size_t order) {
    nodes.push_back(order);     
    node_name_index[name] = nodes.size()-1;
}
 
void TensorNetworkDefinition::AddEdge(std::string name,
                                      std::string node1_name, size_t node1_mode, 
                                      std::string node2_name, size_t node2_mode,
                                      Operation operation) 
{
    // \todo should add error checks

    edges.push_back({
        node_name_index[node1_name],
        node1_mode,
        node_name_index[node2_name],
        node2_mode
    });

    size_t ind = edges.size()-1;
    edge_name_index[name] = ind;

    nonoutput_modes_map.push_back({edges[ind].node1_index, node1_mode});
    nonoutput_modes_map.push_back({edges[ind].node2_index, node2_mode});
}




size_t TensorNetworkDefinition::NumNodes() {
    return nodes.size();
}
// \todo the outputs should be specified ahead of time
size_t TensorNetworkDefinition::OutputOrder() {
    //return output_modes_map.size();
    size_t sum = 0;
    for(size_t i = 0; i < nodes.size(); i++) {
        sum += nodes[i];
    }

    return sum - edges.size() * 2;
}


bool TensorNetworkDefinition::HasOnlyContraction() {
    
    for(size_t i = 0; i < edges.size(); i++) {
        if(edges[i].operation != Operation::CONTRACTION) {
            return false;
        }
    }

    return true;
}



std::vector<std::vector<size_t>> TensorNetworkDefinition::CalcOutputModesMap() {
    
    std::vector<std::vector<size_t>> output_modes_map; 

    for(size_t i = 0; i < nodes.size(); i++) { 
        size_t node_order = nodes[i];
        for(size_t j = 0; j < node_order; j++) {
            bool is_nonoutput_mode = false;
            for(size_t k = 0; k < nonoutput_modes_map.size() && !is_nonoutput_mode; k++) {
                std::vector<size_t> node_mode_pair = nonoutput_modes_map[k];

                if(node_mode_pair[0] == i && node_mode_pair[1] == j) {
                    is_nonoutput_mode = true;
                }
            }
            if(!is_nonoutput_mode) {
                output_modes_map.push_back({i, j});
            }
        }
    }

    return output_modes_map;
}



// \todo remove this
/*
std::vector<std::vector<size_t>> TensorNetworkDefinition::CalcSortedNonOutputModesMap() {
    std::vector<std::vector<size_t>> nonoutput_modes_map;

    // \todo naive for now, possibly won't be a bottleneck though (or even used)
    // probably actually want to do this on a sorted copy of output_modes_map
    for(size_t i = 0; i < nodes.size(); i++) { 
        size_t node_order = nodes[i];
        for(size_t j = 0; j < node_order; j++) {
            bool is_output_mode = false;
            for(size_t k = 0; k < output_modes_map.size() && !is_output_mode; k++) {
                std::vector<size_t> node_mode_pair = output_modes_map[k];

                if(node_mode_pair[0] == i && node_mode_pair[1] == j) {
                    is_output_mode = true;
                }
            }
            if(!is_output_mode) {
                nonoutput_modes_map.push_back({i, j});
            }
        }
    }

    return nonoutput_modes_map;
}
*/

Tensor TensorNetworkDefinition::EvaluateEinsumNaive(const std::vector<Tensor>& tensors) {
    // assume type information has all been checked and the inputs are valid

    // iterate over all output indices, and for each output index iterate over all
    // non-output indices, doing a sum of products.

    std::vector<std::vector<size_t>> output_modes_map = CalcOutputModesMap(); 
    std::vector<size_t> out_tensor_size(OutputOrder());
    for(size_t i = 0; i < out_tensor_size.size(); i++) {
        std::vector<size_t> node_mode_pair = output_modes_map[i];
        out_tensor_size[i] = tensors.at(node_mode_pair[0]).tensor_size[node_mode_pair[1]];
    }
    Tensor out(std::move(out_tensor_size));

    std::vector<size_t> edge_tensor_size(edges.size());
    for(size_t i = 0; i < edge_tensor_size.size(); i++) {
        TensorNetworkEdge edge = edges[i];
        // note the sizes are assumed equal for node1 and node2
        edge_tensor_size[i] = tensors[edge.node1_index].tensor_size[edge.node1_mode]; 
    }
    size_t num_edge_tensor_components = MultiplyElements(edge_tensor_size);

    for(size_t i = 0; i < out.NumComponents(); i++) {
        std::vector<size_t> out_ti = TensorIndex(i, out.tensor_size); 

        float sum = 0;

        for(size_t j = 0; j < num_edge_tensor_components; j++) {
            std::vector<size_t> edge_ti  = TensorIndex(j, edge_tensor_size);
           
            // now break into the tensor indices of the product factors, and multiply
            // prod by the element at each such tensor index
            float prod = 1;
            for(size_t k = 0; k < tensors.size(); k++) {

                std::vector<size_t> k_ti(tensors.at(k).Order());
                for(size_t l = 0; l < output_modes_map.size(); l++) {
                    std::vector<size_t> node_mode_pair = output_modes_map[l];   
                    if(node_mode_pair[0] == k) {
                        k_ti[node_mode_pair[1]] = out_ti[l];
                    }
                }

                for(size_t l = 0; l < edges.size(); l++) {
                    TensorNetworkEdge edge = edges[l];

                    if(edge.node1_index == k) {
                        k_ti[edge.node1_mode] = edge_ti[l];
                    } else if(edge.node2_index == k) {
                        k_ti[edge.node2_mode] = edge_ti[l];
                    }
                }

                prod *= tensors.at(k)[k_ti];
            }
            
            sum += prod;
        }
        
        out[out_ti] = sum; 
    }

    return out;
}

Tensor TensorNetworkDefinition::Evaluate(const std::vector<Tensor>& tensors) {
    // runtime checks to determine the sizes are correct
    // \todo how to handle errors
    // \todo perhaps options to enable / disable runtime checks

    if(NumNodes() != tensors.size()) {
        std::cerr << "Error Evaluate: NumNodes() != tensors.size()\n";
    }

    for(size_t i = 0; i < tensors.size(); i++) {
        if(tensors[i].Order() != nodes[i]) {
            std::cerr << "Error Evaluate: tensor[" << i << "] has the wrong order\n";
        }
    }
    
    for(size_t i = 0; i < edges.size(); i++) {
        TensorNetworkEdge e = edges[i];      
        bool modes_match = tensors[e.node1_index].tensor_size[e.node1_mode] 
                            == tensors[e.node2_index].tensor_size[e.node2_mode];
        if(!modes_match) {
            // \todo better to output the string name of the node
            std::cerr << "Error Evaluate: !modes_match: (node, mode) = " 
                      << "(" << e.node1_index << ", " << e.node1_mode << "), " 
                      << "(" << e.node2_index << ", " << e.node2_mode << ")\n";
        }
    }
   
   
    

    // choose evaluation strategy

    if(HasOnlyContraction()) {
        return EvaluateEinsumNaive(tensors);
    } else {
        // \todo
    }
}

Tensor Evaluate(std::string einsum_definition, 
                std::map<Operation, std::string> operations,
                const std::vector<Tensor>& tensors)
{
    TensorNetworkDefinition network(std::move(einsum_definition), std::move(operations));
                                    
    return network.Evaluate(tensors);
}


} // OPS
