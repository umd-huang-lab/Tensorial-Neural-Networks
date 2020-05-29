#include <vector>
#include <string>


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <dlpack/dlpack.h>


#include "Tensor.h"
#include "TensorNetwork.h"
#include "HelpersTypedefs.h"

namespace OPS {

// this demonstrates the usage of passing in a DLManagedTensor, we end up with a raw pointer to the underlying tensor memory
// Though there's still the problem of figuring out what to do if the tensor data is already on the gpu
float DLManagedTensorCoordinate(py::object pyobject_dlmtensor, size_t index) {
    DLManagedTensor* dlmtensor = (DLManagedTensor *)PyCapsule_GetPointer(pyobject_dlmtensor.ptr(), "dltensor");
    /*
    std::cout << "after cast\n";
	std::cout << "dlmtensor->dl_tensor.data: " << dlmtensor->dl_tensor.data << "\n";
	std::cout << "dlmtensor->dl_tensor.dtype.code: " << dlmtensor->dl_tensor.dtype.code << "\n";
    */ 

    float* data = (float*)dlmtensor->dl_tensor.data;
    return data[index];
}

// \index need to standardize use of dlm_tensor vs dlpack_tensor vs dltensor
std::unique_ptr<Tensor> TNNFromDLPack(py::object pyobject_dlmtensor) {

    DLManagedTensor* dlmtensor 
        = (DLManagedTensor *)PyCapsule_GetPointer(pyobject_dlmtensor.ptr(), "dltensor");

    std::unique_ptr<Tensor> tnn_tensor = std::make_unique<Tensor>(dlmtensor);

    
    return tnn_tensor; 
}

void SpecificExampleTNN(std::vector<py::object> pyobject_dlmtensors) {
    std::vector<std::unique_ptr<Tensor>> tnn_tensors(pyobject_dlmtensors.size());

    for(size_t i = 0; i < tnn_tensors.size(); i++) {
        tnn_tensors[i] = TNNFromDLPack(pyobject_dlmtensors[i]);
    }

    // \todo this is bad (and a trivial fix) but I'm hurrying to get this proof of concept ready
    //       certainly don't want copies, will have to change the signature
    //       of Evalaute, or add one accepting std::vector<unique_ptr<Tensor>>
    std::vector<Tensor> copies(tnn_tensors.size());
    for(size_t i = 0; i < copies.size(); i++) {
        copies[i] = *tnn_tensors[i];
    }


    // this specific example expects 3 tensors
    if(copies.size() != 2) {
        std::cout << "this specific example expects 2 tensors\n";
    }

    TensorNetworkDefinition network; 
    network.AddNode("A", 3);
    network.AddNode("B", 2);
    network.AddEdge("k", {{"A", 0}}, Operation::CONTRACTION, 0); 
    network.AddEdge("i", {{"A", 1}, {"B", 0}}, Operation::CONTRACTION);
    network.AddEdge("j", {{"A", 2}, {"B", 1}}, Operation::CONTRACTION);
    

    // the tensor tensor_sizes must match the specific modes of their correspond node
    
    Tensor out = network.Evaluate(std::move(copies));
    std::cout << "out.FlatString(): " << out.FlatString() << "\n";

}


py::object ConvEinsum(std::vector<std::string> input_subscripts, 
                      std::string output_subscript,
                      std::string convolution_subscript, 
                      std::string subscripts_set,
                      std::vector<py::object> pyobject_dlmtensors)
{

    // \todo this seems to exhibit bugs not exhibited when calling TensorNetworkDefinition::Evaluate elsewhere

    std::vector<std::unique_ptr<Tensor>> tnn_tensors(pyobject_dlmtensors.size());

    for(size_t i = 0; i < tnn_tensors.size(); i++) {
        tnn_tensors[i] = TNNFromDLPack(pyobject_dlmtensors[i]);
        std::cout << "tnn_tensors[" << i << "] = " << tnn_tensors[i]->FlatString() << "\n";
    }


    // \todo this is bad (and a trivial fix) but I'm hurrying to get this proof of concept ready
    //       certainly don't want copies, will have to change the signature
    //       of Evalaute, or add one accepting std::vector<unique_ptr<Tensor>>
    std::vector<Tensor> copies(tnn_tensors.size());
    for(size_t i = 0; i < copies.size(); i++) {
        copies[i] = *tnn_tensors[i];

        std::cout << "copies[" << i << "] = " << copies[i].FlatString() << "\n";
    }

    TensorNetworkDefinition network;
    network.SetConvolutionType(ConvolutionType::SAME);
    for(size_t i = 0; i < input_subscripts.size(); i++) {
        // just name the nodes after their index
        network.AddNode(std::to_string(i), input_subscripts[i].size());
    }

    // \todo can possibly organize things differently and avoid much of this
    for(size_t s = 0; s < subscripts_set.size(); s++) {
        char edge = subscripts_set[s];
        std::vector<TensorNetworkDefinition::NodeMode> edge_parts;
        int output_mode = -1;
        Operation op = Operation::CONTRACTION;

        for(size_t node_index = 0; node_index < input_subscripts.size(); node_index++) {
            for(size_t j = 0; j < input_subscripts[node_index].size(); j++) {
             
                if(edge == input_subscripts[node_index][j]) {
                    edge_parts.push_back({std::to_string(node_index), j});
                }
    
            }
        }

        for(size_t i = 0; i < output_subscript.size(); i++) {
            if(edge == output_subscript[i]) {
                output_mode = int(i);
                break;
            }
        }

        for(size_t i = 0; i < convolution_subscript.size(); i++) {
            if(edge == convolution_subscript[i]) {
                op = Operation::CONVOLUTION; 
                break;
            }
        }

        network.AddEdge(std::string(1, edge), std::move(edge_parts), op, output_mode);
    }
    
     
    Tensor out = network.Evaluate(std::move(copies));
    std::cout << "out.FlatString(): " << out.FlatString() << "\n";
    

    
}

// \todo have to implement everything considered by dlpack_tensor
Tensor::Tensor(DLManagedTensor* dlpack_tensor_in)
    : dlpack_tensor(dlpack_tensor_in),
      tensor_size(dlpack_tensor->dl_tensor.ndim),
      num_components(),
      data((float*)dlpack_tensor->dl_tensor.data), // support more than just float \todo
      managed(false)
{
    for(size_t i = 0; i < tensor_size.size(); i++) {
        tensor_size[i] = dlpack_tensor->dl_tensor.shape[i];
    }
    num_components = (tensor_size.size() == 0 ? 0 : MultiplyElements(tensor_size));
}




PYBIND11_MODULE(tnnlib, m) {
    m.doc() = "tnn plugin"; // optional module docstring
 
    m.def("dlmtensor_coord", &DLManagedTensorCoordinate, "Returns the coordinate of a DLManagedTensor at the given index");

    m.def("specific_example", &SpecificExampleTNN, "Computes a specifc example of a network");

    
    m.def("conv_einsum", &ConvEinsum, "Computes a convolutional tensor network "
                                       "which is represented as a convolutional eisum");


}


} // OPS

