#ifndef TENSOR_H
#define TENSOR_H

/** 
 * Authors: Geoffrey Sangston - gsangsto@umd.edu
 * Huang Group
 *
 * An initial take at define the fundamental tensor operations. Defines
 * contraction, convolution, partial outer product, and outer product. Multiply
 * by matrix can also be achieved as a special case of contraction. 
 *
 */

#include <vector>
#include <string>
#include <memory>

struct DLManagedTensor; 

namespace OPS {

class TensorNetworkDefinition;
struct CPDDecompOut;

// 3rd party library forward declarations, so dependencies are only included by cpp files 



// Just assuming scalars are floats for now.... we can make this a template  
// \todo read this
// https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html
// and see how they handle arbitrary scalar types

// most linear algebra libraries use templates for the dimensions, but this is fine for now
// if this is written well the change should mostly be related to template technicalities, not
// rewriting the code of the algorithms
// probably these implementations will all be naive in a number of ways... they're mostly
// just to get me going... also I'm not sure yet about what variety of tensors actually need
// support

// \todo need to support order 0 tensors
class Tensor {
    public:
        friend class TensorNetworkDefinition;
        Tensor() = default; 
                  //  a tensor without tensor_size is in an 
                  //  unusable state, but people could expect Tensors to behave like
                  //  standard types int or float, this is one reason to prefer
                  //  templating out the tensor_size of the tensor (but that is troublesome
                  //  in its own ways)
                  //
                  // With this they can write Tensor T; and then later T = Contract(A, B);
                  // But they can't construct T in additional steps, they have to copy a new
                  // tensor into it.
                  //
                  // But we might also want to delete this

        /**
         * tensor_size[i] is the size of the ith mode, 
         * tensor_size.size() is the order of the tensor
         */
        explicit Tensor(std::vector<size_t> tensor_size);

        Tensor(DLManagedTensor* dlpack_tensor);

        ~Tensor();

        // \todo look into what Eigen does for copying
        // read https://stackoverflow.com/questions/9322174/move-assignment-operator-and-if-this-rhs
        Tensor(const Tensor& t);
        Tensor& operator=(const Tensor& t);

        // I was using the default move constructor when data was a unique_ptr
        Tensor(Tensor&&);
        

        float& operator[](const std::vector<size_t>& tensor_index);
        const float& operator[](const std::vector<size_t>& tensor_index) const;
        float& operator[](size_t flat_index);
        const float& operator[](size_t flat_index) const;
 

        Tensor operator+(const Tensor& ot);
        Tensor& operator+=(const Tensor& ot);

        Tensor operator-(const Tensor& ot);
        Tensor& operator-=(const Tensor& ot);

        float* Data();


        // Requires the resulting tensor to have the same number of components 
        void Resize(std::vector<size_t> tensor_size_in);

        
        void SetZero();
        void SetUniformRandom(float lower_bound, float upper_bound); 

        /**
         * Swaps the given modes so that T(..., k, ..., l, ...) -> T(..., l, ..., k, ...)
         */
        void SwapAxes(size_t mode_k, size_t mode_l);

        /**
         * SwapAxes mutates the given tensor, whereas SwappedAxes returns a new tensor.
         */
        Tensor SwappedAxes(size_t mode_k, size_t mode_l) const;

        size_t Order() const;
        std::vector<size_t> TensorSize() const;
        size_t NumComponents() const;

        friend bool SameTensorSize(const Tensor& x, const Tensor& y);

        /** 
         * Returns an order (ord(x) + ord(y) - 2)-tensor with elements the inner products of
         * the mode-k fibers of x with the mode-l fibers of Y
         */
        // \todo could template the mode
        friend Tensor Contract(size_t mode_k, size_t mode_l, const Tensor& x, const Tensor& y);
        friend Tensor MultiplyMatrix(size_t mode_k, const Tensor& tensor, const Tensor& matrix);
        friend Tensor Convolve(size_t mode_k, size_t mode_l, const Tensor& x, const Tensor& y);
        friend Tensor PartialOuterProduct(size_t mode_k, size_t mode_l, 
                                          const Tensor& x, const Tensor& y);
        friend Tensor OuterProduct(const Tensor& x, const Tensor& y);

        /**
         * Frobenius Inner Product 
         */
        friend float InnerProduct(const Tensor& x, const Tensor& y);  
        
        friend CPDDecompOut CPDecompALSOrder3(const Tensor& t, size_t rank);

        /**
         * Replaces the matrix at the given slice_index with its pseudo inverse. 
         * (slice_mode1, slice_mode2) specifies the modes which the matrices occupy
         * Baiscally lets you do the pseudo inverse operation in place, without reserving
         * new memory.
         *
         * expects slice_mode1 < slice_mode2.... why though? \todo
         *
         * slice_index is a tensor_index of t, the indices at slice_mode1 and slice_mode2
         * should be 0
         *
         * since the pseudo inverse has the dimensions of the transpose, to do this in place you have to
         * set the transpose, not the pseudo inverse itself 
         */
        void SetPseudoInverseTranspose(size_t slice_mode1, size_t slice_mode2, 
                              std::vector<size_t> slice_index); 
        Tensor CalcPseudoInverse(size_t slice_mode1, size_t slice_mode2,
                                 std::vector<size_t> slice_index);

        std::string FlatString() const;

    private:	

        DLManagedTensor* dlpack_tensor; 

        std::vector<size_t> tensor_size; 
        // \todo is it natural to change the sizes? If so I should have a constructor
        // which accepts just the order and a member SetTensorSize(...)
        // it looks like SetTensorSize would be called Reshape in the literature
        // \todo an empty tensor_size should denote a scalar, and num_components should be 1
 
        size_t num_components;

        // \todo should look into what Eigen does for its memory management

        // I was using unique_ptr for data, with a default destructor for Tensor, 
        // but then realized I need shared_ptr semantics
        // to be interoperable with Pytorch and other libs, except std::shared_ptr in
        // c++14 doesn't implement an array interface, or passing the data pointer
        // as the first element of an array, which makes it painful to work with
        float* data; 
        bool managed = false;
        // kind of hacky to put this here \todo find out what pytorch does
        
}; // Tensor

bool SameTensorSize(const Tensor& x, const Tensor& y);

/**
 * Expects the kth mode of x to have the same size as the lth mode of y.
 * Outputs an order m+n-2 tensor, where m is the order of x and n the order of y.
 */
Tensor Contract(size_t mode_k, size_t mode_l, const Tensor& x, const Tensor& y);


/**
 * Expects matrix to be an order 2 tensor, and that the kth mode of tensor has the same size
 * as the 0th mode of matrixi.
 * Outputs an order m tensor, where m is the order of tensor.
 *
 * Replaces the mode_k vectors v with matrix * v (\todo or is it v * matrix?)
 * Note this isn't directly generalized by Contract because this arranges the output tensor
 * differently. See Table 7 from TNN paper. 
 */
Tensor MultiplyMatrix(size_t mode_k, const Tensor& tensor, const Tensor& matrix);

/**
 * As given by Kolda and Rabanser et al
 * Replaces the mode_k vectors with the scalar dot(vector, v)
 */
Tensor MultiplyVector(size_t mode_k, const Tensor& tensor, const Tensor& vector);

Tensor Convolve(size_t mode_k, size_t mode_l, const Tensor& x, const Tensor& y);
Tensor PartialOuterProduct(size_t mode_k, size_t mode_l, const Tensor& x, const Tensor& y);
Tensor OuterProduct(const Tensor& x, const Tensor& y);
float InnerProduct(const Tensor& x, const Tensor& y);

/**
 * Frobenius Norm
 */
float Norm(const Tensor& x);



/**
 * mat1 and mat2 are order 2 tensors (matrices) with the same number of columns
 */
Tensor KhatriRaoProduct(const Tensor& mat1, const Tensor& mat2);


/**
 * Alternating Least Squares method of computing CP Decomposition in the
 * Order 3 tensor case. Returns an order 3 Tensor cpd such that cpd[:,:,0], cpd[:,:,1], cpd[:,:,2]
 * are the factor matrices of the CP Decomposition, and an order 1 tensor weights, representing
 * the lambda weights.
 */
struct CPDDecompOut {
    Tensor weights;
    Tensor cpd;
};
CPDDecompOut CPDecompALSOrder3(const Tensor& t, size_t rank);



// \todo I wonder if reusing the same vector for index calculations 
//       will give a significant speedup
size_t ColMajorFlatIndex(const std::vector<size_t>& tensor_index, 
                         const std::vector<size_t>& tensor_size);
std::vector<size_t> ColMajorTensorIndex(size_t flat_index, 
                                        const std::vector<size_t>& tensor_size);
size_t ColMajorModeStride(size_t mode, const std::vector<size_t>& tensor_size);

size_t RowMajorFlatIndex(const std::vector<size_t>& tensor_index, 
                         const std::vector<size_t>& tensor_size); 
std::vector<size_t> RowMajorTensorIndex(size_t flat_index, 
                                        const std::vector<size_t>& tensor_size); 
size_t RowMajorModeStride(size_t mode, const std::vector<size_t>& tensor_size);


                        


#define DATA_LAYOUT COL_MAJOR
#define COL_MAJOR 1
#define ROW_MAJOR 2
// a temporary solution \todo
#if DATA_LAYOUT == COL_MAJOR
const auto FlatIndex = ColMajorFlatIndex;
const auto TensorIndex = ColMajorTensorIndex;
const auto ModeStride = ColMajorModeStride;
#elif DATA_LAYOUT == ROW_MAJOR
const auto FlatIndex = RowMajorFlatIndex;
const auto TensorIndex = RowMajorTensorIndex;
const auto ModeStride = RowMajorModeStride;
#endif



/**
 * Returns the initial tensor index of the given mode, where mode_flat_index
 * ranges over [0, num_components / tensor_size[mode_k]], num_components the number
 * of components in the tensor. 
 * Allows easy indexing over the given mode
 */
std::vector<size_t> ModeTensorIndex(size_t mode_k,
                                    size_t mode_flat_index,
                                    const std::vector<size_t>& tensor_size);

// \todo should choose a new name for arbitrary slicing, as Slice appears to refer
// (by Kolda and others) to the case when two indices are sliced
size_t SliceStride(const std::vector<size_t>& slice_modes, 
                   const std::vector<size_t>& tensor_size);

/**
 * Generalizes ModeTensorIndex, which corresponds to passing a single mode to slice_modes
 */
std::vector<size_t> SliceIndex(const std::vector<size_t>& slice_modes,
                                     size_t slice_flat_index,
                                     const std::vector<size_t>& tensor_size);

/** 
 * Slices the modes not in coslice_modes
 */
size_t CoSliceStride(const std::vector<size_t>& coslice_modes,
                     const std::vector<size_t>& tensor_size);
/**
 * Slices the modes not in coslice_modes
 */
std::vector<size_t> CoSliceIndex(const std::vector<size_t>& coslice_modes,
                                     size_t coslice_flat_index,
                                     const std::vector<size_t>& tensor_size);


} // OPS

#endif // TENSOR_H
