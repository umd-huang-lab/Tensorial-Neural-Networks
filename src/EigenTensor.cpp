#ifndef EIGEN_TENSOR
#define EIGEN_TENSOR

#include <iostream>

#include <Eigen/Core>
#include <Eigen/SVD>


#include "Tensor.h"
#include "HelpersTypedefs.h" 


#define DEBUG


/**
 * Putting functions using Eigen api in a separate file for now so others can compile the rest of the
 * code without it for the time being (just move this file from the src
 * directory when compiling, and don't use functions with implementations using Eigen)... 
 * also because it might be a good idea in general
 */

namespace OPS {


void Tensor::SetPseudoInverseTranspose(size_t slice_mode1, size_t slice_mode2, 
                              std::vector<size_t> slice_index) 
{

    #ifdef DEBUG
    if(tensor_size.size() != slice_index.size()) {
        std::cerr << "Error Tensor::SetPseudoInverseTranspose(): tensor_size.size() != slice_index.size()\n";
    }
    #endif

    using namespace Eigen;
    /** 
     * https://eigen.tuxfamily.org/dox/classEigen_1_1Stride.html
     * Stride<OuterStride, InnerStride>
     * OuterStride is an integer specifying the offset between subsequent columns (in a column major
     * layout, or rows in a row major layout), and InnerStride is an integer specifying the offset between
     * subsequent elements
     */

    size_t mode1_stride = ModeStride(slice_mode1, tensor_size);
    size_t mode2_stride = ModeStride(slice_mode2, tensor_size);
    size_t outer_stride = mode2_stride; 
    size_t inner_stride = mode1_stride;
    // our choice of ModeStride handles the determination of the outer/inner stride, this
    // variable relabeling is just to be explicit
    
       
	#if DATA_LAYOUT == COL_MAJOR
    const StorageOptions EIGEN_DATA_LAYOUT = ColMajor; 
	#elif DATA_LAYOUT == ROW_MAJOR
    const StorageOptions EIGEN_DATA_LAYOUT = RowMajor;
    #endif

    Map<MatrixXf, EIGEN_DATA_LAYOUT, Stride<Dynamic,Dynamic>> 
        m(data.get(), tensor_size[slice_mode1], tensor_size[slice_mode2], 
          Stride<Dynamic,Dynamic>(outer_stride, inner_stride)); 


    std::cout << "M:\n" << m << "\n\n"; 

    Matrix<float, Dynamic, Dynamic, EIGEN_DATA_LAYOUT> pseudo_inverse 
        = m.completeOrthogonalDecomposition().pseudoInverse();    

    std::cout << "pseudo_inverse:\n" << pseudo_inverse << "\n\n";

    pseudo_inverse.transposeInPlace();

    size_t num_mat_components = tensor_size[slice_mode1] * tensor_size[slice_mode2];    
    for(size_t i = 0; i < num_mat_components; i++) {
        data[i] = *(pseudo_inverse.data() + i);
    }
}


} // OPS

#endif // EIGEN_TENSOR
