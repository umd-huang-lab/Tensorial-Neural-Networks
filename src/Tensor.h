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

namespace OPS {


// Just assuming scalars are floats for now.... 
// \todo read this
// https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html
// and see how they handle arbitrary scalar types

// most linear algebra libraries use templates for the dimensions, but this is fine for now
// if this is written well the change should mostly be related to template technicalities, not
// rewriting the code of the algorithms
// probably these implementations will all be naive in a number of ways... they're mostly
// just to get me going... also I'm not sure yet about what variety of tensors actually need
// support
class Tensor {
    public:

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
        Tensor(std::vector<size_t> tensor_size);
        ~Tensor() = default;

        // \todo look into what Eigen does for copying
        // read https://stackoverflow.com/questions/9322174/move-assignment-operator-and-if-this-rhs
        Tensor(const Tensor& t);
        Tensor& operator=(const Tensor& t);

        Tensor(Tensor&&) = default;
        



        float& operator[](const std::vector<size_t>& tensor_index);
        float& operator[](size_t flat_index);

        Tensor operator+(const Tensor& ot);
        Tensor& operator+=(const Tensor& ot);

        // Requires the resulting tensor to have the same number of components 
        void Resize(std::vector<size_t> tensor_size_in);
        void SetZero();

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
        friend Tensor Convolve(size_t mode_k, size_t mode_l, const Tensor& x, const Tensor& y);
        friend Tensor PartialOuterProduct(size_t mode_k, size_t mode_l, 
                                          const Tensor& x, const Tensor& y);
        friend Tensor OuterProduct(const Tensor& x, const Tensor& y);

        /**
         * Following Kolda
         */
        friend float InnerProduct(const Tensor& x, const Tensor& y); 

        std::string FlatString() const;

    private:	


        std::vector<size_t> tensor_size; 
        // \todo is it natural to change the sizes? If so I should have a constructor
        // which accepts just the order and a member SetTensorSize(...)
        // it looks like SetTensorSize would be called Reshape in the literature
 
        size_t num_components;

        // \todo should look into what Eigen does for its memory management
        std::unique_ptr<float[]> data; 
        
        

}; // Tensor

bool SameTensorSize(const Tensor& x, const Tensor& y);

Tensor Contract(size_t mode_k, size_t mode_l, const Tensor& x, const Tensor& y);
Tensor Convolve(size_t mode_k, size_t mode_l, const Tensor& x, const Tensor& y);
Tensor PartialOuterProduct(size_t mode_k, size_t mode_l, const Tensor& x, const Tensor& y);
Tensor OuterProduct(const Tensor& x, const Tensor& y);
float InnerProduct(const Tensor& x, const Tensor& y);

size_t ColMajorFlatIndex(const std::vector<size_t>& tensor_index, 
                         const std::vector<size_t>& tensor_size);

std::vector<size_t> ColMajorTensorIndex(size_t flat_index, const std::vector<size_t>& tensor_size);
size_t ColMajorModeStride(size_t mode, const std::vector<size_t>& tensor_size);

size_t RowMajorFlatIndex(const std::vector<size_t>& tensor_index, 
                         const std::vector<size_t>& tensor_size); 
std::vector<size_t> RowMajorTensorIndex(size_t flat_index, const std::vector<size_t>& tensor_size); 
size_t RowMajorModeStride(size_t mode, const std::vector<size_t>& tensor_size);




// You have to select all three in a group and none in the other
//const auto FlatIndex = ColMajorFlatIndex;
//const auto TensorIndex = ColMajorTensorIndex;
//const auto ModeStride = ColMajorModeStride;

const auto FlatIndex = RowMajorFlatIndex;
const auto TensorIndex = RowMajorTensorIndex;
const auto ModeStride = RowMajorModeStride;

} // OPS

#endif // TENSOR_H
