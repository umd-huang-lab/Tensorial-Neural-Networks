#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <dlpack/dlpack.h>



#include "Tensor.h"



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


PYBIND11_MODULE(tnn, m) {
    m.doc() = "tnn plugin"; // optional module docstring
 
    m.def("dlmtensor_coord", &DLManagedTensorCoordinate, "Returns the coordinate of a DLManagedTensor at the given index");
}

