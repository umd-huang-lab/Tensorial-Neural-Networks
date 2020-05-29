#include <iostream>
#include <algorithm>
#include <utility>


#include "TensorNetwork.h"
#include "Tensor.h"
#include "HelpersTypedefs.h"

namespace OPS {



TensorNetworkDefinition::TensorNetworkDefinition(std::string einsum_definition, 
                                                 std::map<Operation, std::string> operations) 
{

}

void TensorNetworkDefinition::SetConvolutionType(ConvolutionType type) {
    convolution_type = type; 
}



void TensorNetworkDefinition::AddNode(std::string name, size_t order) {
    nodes.push_back(order);     
    node_name_index[name] = nodes.size()-1;
}



void TensorNetworkDefinition::AddEdge(std::string edge_name,
                                      std::vector<NodeMode> edge_parts,
                                      Operation operation,
                                      int output_mode)
{

    // \todo add error checks enforcing tensor network invariants like one edge per mode
    // \todo right now the entire network must be specified, no edges / modes are 
    //       implicitly determined. If I go with that, should add error checks enforcing
    //       this    

    std::vector<InternalEdgePart> internal_edge_parts;
    internal_edge_parts.reserve(edge_parts.size());
    for(size_t i = 0; i < edge_parts.size(); i++) {
        InternalEdgePart part = {
            node_name_index[edge_parts[i].node_name],
            edge_parts[i].mode,
            //edge_parts[i].operation
            operation
        };
        internal_edge_parts.push_back(part);
    }
    
    
    edges.push_back({std::move(internal_edge_parts), output_mode});

    size_t ind = edges.size()-1;
    edge_name_index[edge_name] = ind;

    if(output_mode >= 0) {
		// insert in order
        OutputMode om; om.output_mode = output_mode;
		auto insert_pos = std::upper_bound(output_modes.begin(), output_modes.end(),
										   om, // \todo this is strange 
										   [](const OutputMode& a, const OutputMode& b) {
												return a.output_mode < b.output_mode;
										   });
		output_modes.insert(insert_pos, {ind, size_t(output_mode)});
    }
	
	// \todo not sure if I still need this
    if(output_mode < 0) {
        for(size_t i = 0; i < edge_parts.size(); i++) {
            nonoutput_modes_map.push_back({
                node_name_index[edge_parts[i].node_name],
                edge_parts[i].mode
            });
        }
    }
}



void TensorNetworkDefinition::AddMultiOperationEdge(std::string edge_name,
                                              std::vector<EdgePart> edge_parts,
                                              int output_mode)
{

}




size_t TensorNetworkDefinition::NumNodes() {
    return nodes.size();
}

bool TensorNetworkDefinition::HasOnlyContraction() {
    
    for(size_t i = 0; i < edges.size(); i++) {
        if(!edges[i].HasOnlyContraction()) {
            return false;
        }
    }

    return true;
}



bool TensorNetworkDefinition::HasNoMultiOperationEdges() {
     
    for(size_t i = 0; i < edges.size(); i++) {
        if(!edges[i].HasSingleOperation()) {
            return false;
        }
    }

    return true;           
}




// \todo might consider inlining this 
std::vector<size_t> 
TensorNetworkDefinition::CalcOutputTensorSize(const std::vector<Tensor>& tensors) {
    std::vector<size_t> output_tensor_size(output_modes.size());
 
	for(size_t i = 0; i < output_modes.size(); i++) {
        OutputMode& o = output_modes[i];
        if(o.is_edge) {
            size_t edge_node0_index = edges[o.index].edge_parts[0].node_index;
            size_t edge_node0_mode = edges[o.index].edge_parts[0].mode;
          
            output_tensor_size[i] = tensors[edge_node0_index].tensor_size[edge_node0_mode]; 
        } else {
            output_tensor_size[i] = tensors[o.index].tensor_size[o.node_mode];
        }
	} 

    return output_tensor_size;
}

/*
Tensor TensorNetworkDefinition::EvaluateEinsumNaive(const std::vector<Tensor>& tensors) {
    // assume type information has all been checked and the inputs are valid

    // iterate over all output indices, and for each output index iterate over all
    // non-output indices, doing a sum of products.
    // non-output indices correspond to operation edges

    
    std::vector<size_t> out_tensor_size(output_modes.size()); 
   
    for(size_t i = 0; i < out_tensor_size.size(); i++) {
        NodeIndexMode index_mode = output_modes_map[i];
        out_tensor_size[i] = tensors.at(index_mode.index).tensor_size[index_mode.mode];
    }
    Tensor out(std::move(out_tensor_size));

    std::vector<size_t> edge_tensor_size(edges.size());
    for(size_t i = 0; i < edge_tensor_size.size(); i++) {
        std::vector<InternalEdgePart>& edge_parts = edges[i].edge_parts;
        size_t node_index = edge_parts[0].node_index; 
        size_t node_mode = edge_parts[0].mode;
        // note the mode sizes are assumed equal for all the nodes in an edge
        edge_tensor_size[i] = tensors[node_index].tensor_size[node_mode]; 
    }
    size_t num_edge_tensor_components = MultiplyElements(edge_tensor_size);
    
    for(size_t i = 0, outNumComponents = out.NumComponents(); i < outNumComponents; i++) {
        std::vector<size_t> out_ti = TensorIndex(i, out.tensor_size); 

        float sum = 0;
        for(size_t j = 0; j < num_edge_tensor_components; j++) {
            std::vector<size_t> edge_ti  = TensorIndex(j, edge_tensor_size);
           
            // now break into the tensor indices of the product factors, and multiply
            // prod by the element at each such tensor index
            float prod = 1;
            for(size_t k = 0; k < tensors.size(); k++) {

                // k_ti is a tensor index for the kth tensor
                std::vector<size_t> k_ti(tensors.at(k).Order());
                for(size_t l = 0; l < output_modes_map.size(); l++) {
                    NodeIndexMode index_mode = output_modes_map[l];   
                    if(index_mode.index == k) {
                        k_ti[index_mode.mode] = out_ti[l];
                    }
                }

                for(size_t l = 0; l < edges.size(); l++) {
                    std::vector<InternalEdgePart>& edge_parts = edges[l].edge_parts;
                  
                    for(size_t m = 0; m < edge_parts.size(); m++) {
                        if(edge_parts[m].node_index == k) {
                            k_ti[edge_parts[m].mode] = edge_ti[l];
                        }
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
*/





Tensor TensorNetworkDefinition::EvaluateConvNaive(const std::vector<Tensor>& tensors) {
 

    // assume type information has all been checked and the inputs are valid

    // \todo rename "operation" to something evoking the summation_indices in the forall sum
    // \todo could try to precompute some of this setup, (the parts not using tensors directly)
    //       it depends on how the user wants to call this.
    // \todo there's some code duplication, could think about factoring things out, or better,
    //       reorganizing the code paths so eliminate separate paths doing the same thing 
    std::vector<size_t> out_tensor_size = CalcOutputTensorSize(tensors);   
    std::cout << "out_tensor_size: " << out_tensor_size << "\n";
    Tensor out(std::move(out_tensor_size));

    
    std::vector<size_t> op_ind2edge_ind_map; 
    for(size_t i = 0; i < edges.size(); i++) { 
        if(edges[i].output_mode < 0) {
            op_ind2edge_ind_map.push_back(i);
        }
    } 
    
    std::vector<size_t> operation_tensor_size(op_ind2edge_ind_map.size()); 
    for(size_t i = 0; i < operation_tensor_size.size(); i++) {
        size_t edge_ind = op_ind2edge_ind_map[i];

        size_t node_ind = edges[edge_ind].edge_parts[0].node_index;
        size_t node_mode = edges[edge_ind].edge_parts[0].mode;
        operation_tensor_size[i] = tensors[node_ind].tensor_size[node_mode]; 
    }
    size_t num_operation_tensor_components = MultiplyElements(operation_tensor_size);
    // MultiplyElements returns 1 if operation_tensor_size is empty, so the loop will
    // run at least one time

    std::vector<size_t> conv_parts_tensor_size;
    std::vector<std::pair<size_t, size_t>> conv_parts_intervals;

    
    for(size_t i = 0; i < edges.size(); i++) {
        if(edges[i].HasOnlyConvolution()) {
            size_t first = conv_parts_tensor_size.size();
            size_t node_ind = edges[i].edge_parts[0].node_index;
            size_t node_mode = edges[i].edge_parts[0].mode;
            size_t mode_size = tensors[node_ind].tensor_size[node_mode];

            // the number of implicit indices is one less than the number of edge parts
            for(size_t j = 0; j < edges[i].edge_parts.size()-1; j++) {
                conv_parts_tensor_size.push_back(mode_size);
            }

            size_t last = conv_parts_tensor_size.size();
            conv_parts_intervals.push_back(std::pair<size_t, size_t>(first, last));

        } else { // not supporting mixed edges yet
            conv_parts_intervals.push_back(std::pair<size_t, size_t>(0, 0));
            // want conv_parts_intervals to correspond to edges
        }
    }
    size_t num_conv_parts_tensor_components = MultiplyElements(conv_parts_tensor_size);
    // if conv_parts_tensor_size is empty then MultiplyElements returns 1 and so the loop
    // will run at least once

    ////////////////////////////////////////////////
    // Initialize kth_tensor_modes_info
    // This can be computed directly from the einsum description

    
    // OUT is a mode appearing to the right of the arrow in an einsum (parameterized by the
    // for all symbol),
    // OPERATION is an explicit mode not appearing to the right of the arrow 
    // (parameterized by the summation symbol and not an implicit convolution index)
    // IMPLICIT is an implicit convolution index
    enum class ModeType { OUT, OPERATION, IMPLICIT };

    /**
     *  this struct describes the formal data of a subscript
     *  index into output_modes array (OUT), op_ind2edge_ind_map (OPERATION) 
     *  or edges (IMPLICIT) 
     */
    struct ModeInfo {
        ModeType mode_type;   
        size_t index; 
        size_t implicit_offset;
    };

    // calculate for each k a vector containing structs which specify the formal
    // subscript for that given mode
    // the following vector of vectors essentially encapsulates the input from einsum
    std::vector<std::vector<ModeInfo>> kth_tensor_modes_info(nodes.size());
    for(size_t k = 0; k < nodes.size(); k++) {
        kth_tensor_modes_info[k].resize(nodes[k]);       
    }


    {
        for(size_t l = 0; l < output_modes.size(); l++) {
            OutputMode om = output_modes[l];
            if(om.is_edge) {
                Operation single_op;
                if(edges[om.index].HasSingleOperation(&single_op)) {
                std::vector<InternalEdgePart>& edge_parts = edges[om.index].edge_parts; 

                if(single_op == Operation::CONVOLUTION) {    
                    size_t m = 0;
                    for(; m < edge_parts.size()-1; m++) {
                        size_t k = edge_parts[m].node_index;
                        kth_tensor_modes_info[k][edge_parts[m].mode] 
                            = {ModeType::IMPLICIT, om.index, m};
                    }

                    size_t k = edge_parts[m].node_index;
                    kth_tensor_modes_info[k][edge_parts[m].mode] = {ModeType::OUT, l};

                } else { // Operation::CONTRACTION
                    for(size_t m = 0; m < edge_parts.size(); m++) {
                        size_t k = edge_parts[m].node_index;
                        kth_tensor_modes_info[k][edge_parts[m].mode] = {ModeType::OUT, l};  
                    }
                }

                } else {
                    std::cerr << "Not supporting multiedges yet\n";
                }
            } else {
                kth_tensor_modes_info[om.index][om.node_mode] = {ModeType::OUT, l};
            }
        }
    }

    for(size_t l = 0; l < op_ind2edge_ind_map.size(); l++) {
        
        size_t edge_ind = op_ind2edge_ind_map[l];
        std::vector<InternalEdgePart>& edge_parts = edges[edge_ind].edge_parts; 
        size_t cur_conv_part = 0;
        size_t num_conv_parts = edges[l].NumConvolutionParts();
        

        for(size_t m = 0; m < edge_parts.size(); m++) {
            size_t k = edge_parts[m].node_index;

            if(edge_parts[m].operation == Operation::CONTRACTION
               || cur_conv_part == num_conv_parts-1) 
            {
                // \todo think about renaming operation/implicit
                kth_tensor_modes_info[k][edge_parts[m].mode] 
                    = {ModeType::OPERATION, l}; 
            } else { 
                
                kth_tensor_modes_info[k][edge_parts[m].mode]
                    = {ModeType::IMPLICIT, edge_ind, m};
            }
 
            if(edge_parts[m].operation == Operation::CONVOLUTION) {
                cur_conv_part++;
            }                       
                
            kth_tensor_modes_info[k][edge_parts[m].mode] = {ModeType::OPERATION, l};  
        }
        
    }

    
   
     
    // Initialize kth_tensor_modes_info 
    ////////////////////////////////////////////////
   



    //// what follows is the main computation, the above is in preparation for this
    // run the loop at least once to handle when the output tensor is a scalar
    // \todo it would be preferable to separate out the different convolution types

    for(size_t i = 0, outNumComponents = out.NumComponents();
        i < outNumComponents || i == 0; i++) 
    {
    std::vector<size_t> out_ti = TensorIndex(i, out.tensor_size);
        // the components of out_ti correspond to the elements of output_modes

    
    float sum = 0;
    for(size_t j = 0; j < num_operation_tensor_components; j++) {
    std::vector<size_t> op_ti = TensorIndex(j, operation_tensor_size); 
    // the components of op_ti corresponds to the elements of edges
    // actually this is not true anymore, since edges can have output modes
    // as stands, corresponds to the elements of edges which don't have output modes
    // you can index the jth "edge without output mode" by 
    // edge[op_ind2edge_ind_map[j]]

    for(size_t j1 = 0; j1 < num_conv_parts_tensor_components; j1++) {
    std::vector<size_t> conv_parts_ti = TensorIndex(j1, conv_parts_tensor_size);    
    // the components of conv_parts_ti correspond to the elements of the array obtained
    // by appending in order the convolution edge parts of each edge, except for 
    // the last convolution part of each edge.
    
    // you need to iterate over Operation indices x Implicit indices, so it makes
    // sense that there could be a multiplicative, double, for-loop 
    
        float prod = 1;
        for(size_t k = 0; k < tensors.size(); k++) {
            // k_ti is a tensor index for the kth tensor
            std::vector<size_t> k_ti(tensors.at(k).Order(), 0);
            bool index_is_out_of_range = false;
            std::vector<ModeInfo>& modes_info = kth_tensor_modes_info[k];

            for(size_t l = 0; l < k_ti.size() && !index_is_out_of_range; l++) {
                // calculate the index at the lth mode of the kth tensor
                // \todo is it possible to remove these conditional branches from
                //       this innermost loop?

                ModeInfo mode_info = modes_info[l];
                switch(mode_info.mode_type) {
                case ModeType::OUT: {
                    OutputMode output_mode_info = output_modes[mode_info.index];
                    if(output_mode_info.is_edge) {
                        size_t edge_ind = output_mode_info.index;
                        Edge& output_mode_edge = edges[edge_ind];

                        Operation single_op;
                        if(output_mode_edge.HasSingleOperation(&single_op)) {
                            if(single_op == Operation::CONVOLUTION) {
                                // \todo it's unfortunate that this duplicates code b/w
                                // the cases that the index is an output, and when it is not
                                // of the form i - i1 - i2 - i3
                                
                                 size_t mode_size = output_mode_edge.GetModeSize(tensors, l);
                                 size_t out_ind = out_ti[output_mode_info.output_mode];

                                std::pair<size_t, size_t> 
                                interval = conv_parts_intervals[edge_ind]; 

                                size_t conv_ind_sum = 0;
                                for(size_t cis = interval.first; cis < interval.second; cis++) {
                                    conv_ind_sum += conv_parts_ti[cis];
                                }
                                                            
                                switch(convolution_type) {
                                case ConvolutionType::CYCLIC: {
                                    k_ti[l] = PModi(out_ind - conv_ind_sum, mode_size);
                                } break;
                                case ConvolutionType::SAME: {
 
                                size_t conv_offset_ind= output_mode_edge.GetSAMEOffset(tensors); 
                                k_ti[l] = out_ind + conv_offset_ind - conv_ind_sum;

                                if(out_ind < conv_ind_sum || k_ti[l] >= mode_size) {
                                    index_is_out_of_range = true; 
                                }

                                } break;
                                }
                            } else { // Operation::CONTRACTION
                                // of the form i
                                k_ti[l] = out_ti[output_mode_info.output_mode];
                            }
                        } else {
                            // \todo
                            std::cerr << "Multiedges not implemented\n";
                        }
                    } else { // output_mode is node
                        k_ti[l] = out_ti[output_mode_info.output_mode];
                    }
                } break; 
                case ModeType::OPERATION: {
                                    
                    size_t op_ti_index = mode_info.index;
                    size_t edge_ind = op_ind2edge_ind_map[op_ti_index];
                    Edge& edge = edges[edge_ind]; 
                    Operation single_op;
                    if(edge.HasSingleOperation(&single_op)) {
                        if(single_op == Operation::CONVOLUTION) { 
                            // of the form i - i1 - i2 - i3
                            size_t mode_size = edge.GetModeSize(tensors, l);
                            size_t ind = op_ti[op_ti_index];

                            std::pair<size_t, size_t> 
                            interval = conv_parts_intervals[edge_ind];

                            // \todo a subroutine for this inner forloop possibly,
                            // make sure it gets inlined
                            size_t conv_ind_sum = 0;
                            for(size_t cis = interval.first; cis < interval.second; cis++) {
                                conv_ind_sum += conv_parts_ti[cis];
                            }

                            switch(convolution_type) {
                            case ConvolutionType::CYCLIC: {
                                k_ti[l] = PModi(ind - conv_ind_sum, mode_size);
                                
                            } break;
                            case ConvolutionType::SAME: { 
                               
                                size_t conv_offset_ind = edge.GetSAMEOffset(tensors); 
                                k_ti[l] = ind + conv_offset_ind - conv_ind_sum;
                                if(ind < conv_ind_sum || k_ti[l] >= mode_size) {
                                    index_is_out_of_range = true; 
                                }
                            } break;
                            
                            }
                            
                        } else { // Operation::CONTRACTION 
                            k_ti[l] = op_ti[op_ti_index];
                        }
                    } else {
                        // \todo
                        std::cerr << "Multiedges not implemented\n";
                    }
                } break;
                case ModeType::IMPLICIT: {
                    size_t edge_ind = mode_info.index;
                    size_t conv_ind_start = conv_parts_intervals[edge_ind].first;
                    k_ti[l] = conv_parts_ti[conv_ind_start + mode_info.implicit_offset]; 
                } break;
                }

            }
          
            if(index_is_out_of_range) { 
                prod *= 0;
                break;
            } else {
                prod *= tensors.at(k)[k_ti];  
                
            }
        }
        sum += prod;
    } // j1
    } // j
        
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
          
        bool modes_match = true;
        InternalEdgePart edge_part = edges[i].edge_parts[0];
        size_t mode_size = tensors[edge_part.node_index].tensor_size[edge_part.mode];

        for(size_t j = 1; j < edges[i].edge_parts.size() && modes_match; j++) {
            edge_part = edges[i].edge_parts[j]; 
            if(edge_part.operation == Operation::CONTRACTION) {
                // convolution modes can be different sizes
                modes_match &= 
                   (mode_size == tensors[edge_part.node_index].tensor_size[edge_part.mode]);
            }
        }

        if(!modes_match) {
            // \todo better to output the string name of the node
            std::cerr << "Error Evaluate: !modes_match\n"; 
                      //<< "(" << e.node1_index << ", " << e.node1_mode << "), " 
                      //<< "(" << e.node2_index << ", " << e.node2_mode << ")\n";
        }
    }
   
   
    // choose evaluation strategy

    if(HasOnlyContraction()) {
        //return EvaluateEinsumNaive(tensors);
    } else if(HasNoMultiOperationEdges()) { 
        return EvaluateConvNaive(tensors);
    } else {
        // \todo
        
    }
}




bool TensorNetworkDefinition::Edge::HasSingleOperation(Operation* single_operation_out) {
    Operation first_op = edge_parts[0].operation;
    for(size_t i = 1; i < edge_parts.size(); i++) {
        if(edge_parts[i].operation != first_op) {
            return false;
        }
    }
    if(single_operation_out != nullptr) {
        *single_operation_out = first_op; 
    }
    return true;
}

bool TensorNetworkDefinition::Edge::HasOnlyContraction() {
    Operation op;
    bool single = HasSingleOperation(&op);
    
    return single && op == Operation::CONTRACTION;
}

bool TensorNetworkDefinition::Edge::HasOnlyConvolution() {
    Operation op;
    bool single = HasSingleOperation(&op);
    
    return single && op == Operation::CONVOLUTION;
}

// probably want to store this instead of computing it in an inner loop
size_t TensorNetworkDefinition::Edge::NumConvolutionParts() {
    size_t num_convolution_parts = 0;
    for(size_t i = 0; i < edge_parts.size(); i++) {
        if(edge_parts[i].operation == Operation::CONVOLUTION) {
            num_convolution_parts++;
        }
    }
    return num_convolution_parts;
}

size_t TensorNetworkDefinition::Edge::GetModeSize(const std::vector<Tensor>& tensors,
                                                  size_t edge_part) 
{
    return tensors[edge_parts[edge_part].node_index].tensor_size[edge_parts[edge_part].mode];
}

void TensorNetworkDefinition::Edge::GetMinMaxModeSizes(const std::vector<Tensor>& tensors,
                                                       size_t& min, size_t& max) 
{
    static size_t temp_min = tensors[edge_parts[0].node_index].tensor_size[edge_parts[0].mode];
    static size_t temp_max = temp_min;

    for(size_t i = 1; i < tensors.size(); i++) {
        size_t mode_size = tensors[edge_parts[i].node_index].tensor_size[edge_parts[i].mode];

        if(mode_size < temp_min) {
            temp_min = mode_size;
        } else if(mode_size > temp_max) {
            temp_max = mode_size;
        }
    } 

    min = temp_min;
    max = temp_max; 
}

size_t TensorNetworkDefinition::Edge::GetSAMEOffset(const std::vector<Tensor>& tensors) {
    // could improve this method by removing the call to GetMinMaxModeSizes
    size_t min_mode_size;
    size_t max_mode_size; 
    this->GetMinMaxModeSizes(tensors, min_mode_size, max_mode_size);

    /**
     * We do this complicated sum because we want to support multiway convolution. This sum
     * computes the padding that results from binary SAME convolutions computed left to right,
     * as in (((A * B) * C) * D) * E ...
     * Each binary SAME chooses the padding to be (N-1)/2, where N is 
     * the lesser of the two mode sizes. This padding indexes an element of the full 
     * convolution. For GetSAMEOffset we effectively index an element of the full convolution
     * resulting from all of the binary convolutions.
     */
    size_t conv_offset_ind = 0;
    for(size_t m = 0; m < tensors.size(); m++) {
        conv_offset_ind 
        += (tensors[edge_parts[m].node_index].tensor_size[edge_parts[m].mode]-1)/2;
    }
    conv_offset_ind -= (max_mode_size-1)/2;

    return conv_offset_ind;
}


} // OPS
