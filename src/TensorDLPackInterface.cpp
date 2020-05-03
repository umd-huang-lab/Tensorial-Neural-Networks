#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <dlpack/dlpack.h>


#include "Tensor.h"
#include "TensorNetwork.h"
#include "HelpersTypedefs.h"

namespace OPS {

// this demonstrates the correctness of passing in a DLManagedTensor, we end up with a raw pointer to the underlying tensor memory
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

// \index need to standardzed use of dlm_tensor vs dlpack_tensor vs dltensor
std::unique_ptr<Tensor> TNNFromDLPack(py::object pyobject_dlmtensor) {

    DLManagedTensor* dlmtensor 
        = (DLManagedTensor *)PyCapsule_GetPointer(pyobject_dlmtensor.ptr(), "dltensor");

    std::unique_ptr<Tensor> tnn_tensor = std::make_unique<Tensor>(dlmtensor);

    std::cout << "created tensor with tensor_size = " << tnn_tensor->TensorSize() << "\n";
    return tnn_tensor; 
}

void SpecificExampleTNN(std::vector<py::object> pyobject_dlmtensors) {
    std::vector<std::unique_ptr<Tensor>> tnn_tensors(pyobject_dlmtensors.size());

    for(size_t i = 0; i < tnn_tensors.size(); i++) {
        tnn_tensors[i] = TNNFromDLPack(pyobject_dlmtensors[i]);
    }

    // \todo this is bad but I'm hurrying to get this proof of concept ready
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




PYBIND11_MODULE(tnn, m) {
    m.doc() = "tnn plugin"; // optional module docstring
 
    m.def("dlmtensor_coord", &DLManagedTensorCoordinate, "Returns the coordinate of a DLManagedTensor at the given index");

    m.def("specific_example", &SpecificExampleTNN, "Computes a specifc example of a network");


}


} // OPS

