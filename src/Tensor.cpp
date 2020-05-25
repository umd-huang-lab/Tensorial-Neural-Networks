#include <iostream>
#include <sstream>
#include <cmath>
#include <random>
#include <utility>
#include <memory>

#include "Tensor.h"
#include "HelpersTypedefs.h"

namespace OPS {




Tensor::Tensor(std::vector<size_t> tensor_size_in) 
    : tensor_size(std::move(tensor_size_in)), 
      num_components(tensor_size.size() == 0 ? 0 : MultiplyElements(tensor_size)),
      data(new float[NumComponents()]),
      managed(true) 
{}



Tensor::Tensor(const Tensor& t) {
    
    tensor_size = t.tensor_size;
    num_components = t.num_components;
    data = new float[t.num_components];  
    for(size_t i = 0; i < t.num_components; i++) {
        data[i] = t.data[i]; 
    }
    managed = true;
}

Tensor& Tensor::operator=(const Tensor& t) {
    
    data = new float[t.num_components]; 
    for(size_t i = 0; i < t.num_components; i++) {
        data[i] = t.data[i]; 
    }
    managed = true;

    num_components = t.num_components;
    tensor_size = t.tensor_size;
    return *this;
}
/*
// alternative
Tensor& Tensor::operator=(Tensor&& t) {
    tensor_size = std::move(t.tensor_size);
    num_components = t.num_components;
    data = std::move(t.data); 
    return *this;
}
*/


Tensor::Tensor(Tensor&& t) {
    tensor_size = std::move(t.tensor_size); 
    num_components = t.num_components;

    data = t.data;
    managed = true;

    t.data = nullptr;
    t.managed = false;
}

Tensor::~Tensor() { 
   
    if(managed) {
        delete[] data;
    }
    // otherwise the platform that we borrowed the memory from is responsible for
    // deleting
}



Tensor Tensor::operator+(const Tensor& ot) {
    Tensor out = *this;
    out += ot; 
    return out;
} 

Tensor& Tensor::operator+=(const Tensor& ot) {
	#ifdef DEBUG
    if(!SameTensorSize(*this, ot)) {
        std::cerr << "Error Tensor::operator+= !SameTensorSize(*this, ot)\n";
    }
    #endif // DEBUG	
    for(size_t i = 0, N = num_components; i < N; i++) {
        data[i] += ot.data[i];
    }

	return *this;	
}



Tensor Tensor::operator-(const Tensor& ot) {
    Tensor out = *this;
    out -= ot; 
    return out;
} 

Tensor& Tensor::operator-=(const Tensor& ot) {
	#ifdef DEBUG
    if(!SameTensorSize(*this, ot)) {
        std::cerr << "Error Tensor::operator-= !SameTensorSize(*this, ot)\n";
    }
    #endif // DEBUG	
    for(size_t i = 0, N = num_components; i < N; i++) {
        data[i] -= ot.data[i];
    }

	return *this;	
}





/**
 * The inverse of ColMajorTensorIndex
 */
// it seems like FlatIndex should/could be a member method
size_t ColMajorFlatIndex(const std::vector<size_t>& tensor_index, 
                         const std::vector<size_t>& tensor_size) 
{
    #ifdef DEBUG
    if(tensor_index.size() != tensor_size.size()) {
        std::cerr << "Error FlatIndex: tensor_index.size() != tensor_size.size() "
                  << "tensor_index.size() == " << tensor_index.size() 
                  << ", tensor_size.size() == " << tensor_size.size() << "\n";
    }
    #endif // DEBUG
    
    size_t ret = 0;
    size_t offset = 1;
    for(size_t i = 0, N = tensor_index.size(); i < N; i++) {
        ret += tensor_index[i]*offset;
        offset *= tensor_size[i];
    }
    return ret;
}

/**
 * The inverse of ColMajorFlatIndex
 */
// \todo could make a member function version
std::vector<size_t> ColMajorTensorIndex(size_t flat_index, const std::vector<size_t>& tensor_size) {
    #ifdef DEBUG
    if(tensor_size.size() == 0) {
        std::cerr << "Error TensorIndex: tensor_size.size() == 0\n";
    }
    #endif // DEBUG

    std::vector<size_t> tensor_index(tensor_size.size());
    size_t offset = MultiplyElements(tensor_size);
    // \todo there may be a way to arrange this to skip the up front multiply

    for(int i = int(tensor_size.size()) - 1; i >= 0; i--) {
        offset /= tensor_size[i];  
        tensor_index[i] = flat_index/offset;
        flat_index -= tensor_index[i] * offset; // remainder        
    }

    return tensor_index;
}

/**
 * \todo "RowMajor" or "ColMajor" are only the right name in terms of rank 2 tensors (matrices)
 *       Should choose the proper name.... the paper 
 *       "Tensorial Neural Networks: Generalization of Neural Networks 
 *        and Application to Model Compression" 
 *       is using the "RowMajor" convention, see section B page 10
 *
 * see https://en.wikipedia.org/wiki/Row-_and_column-major_order#Address_calculation_in_general
 */
size_t RowMajorFlatIndex(const std::vector<size_t>& tensor_index, 
                         const std::vector<size_t>& tensor_size) 
{
    size_t ret = 0;
    size_t offset = 1;
    for(int i = int(tensor_index.size())-1; i >= 0; i--) {
        ret += tensor_index[i]*offset;
        offset *= tensor_size[i];
    }
    return ret;
}

std::vector<size_t> RowMajorTensorIndex(size_t flat_index, const std::vector<size_t>& tensor_size) {
    size_t N = tensor_size.size();
    std::vector<size_t> tensor_index(N); 
    size_t offset = MultiplyElements(tensor_size);

    for(size_t i = 0; i < N; i++) {
        offset /= tensor_size[i];
        tensor_index[i] = flat_index/offset;
        flat_index -= tensor_index[i] * offset;
    }

    return tensor_index;
}

/**
 * Returns the offset to increment along a given mode, in the "ColMajor" convention
 */
size_t ColMajorModeStride(size_t mode, const std::vector<size_t>& tensor_size) {
    size_t offset = 1; 
    for(size_t j = 0; j < mode; j++) { 
        offset *= tensor_size[j];
    }
    return offset;
}


/**
 * Returns the offset to increment along a given mode, in the "RowMajor" convention
 */
size_t RowMajorModeStride(size_t mode, const std::vector<size_t>& tensor_size) {
    size_t offset = 1; 
    for(size_t j = mode+1, N = tensor_size.size(); j < N; j++) { 
        offset *= tensor_size[j];
    }
    return offset;
}

std::vector<size_t> ModeTensorIndex(size_t mode_k,
                                    size_t mode_flat_index,
                                    const std::vector<size_t>& tensor_size)
{
    std::vector<size_t> mode_tensor_size; mode_tensor_size.reserve(tensor_size.size()-1);
    mode_tensor_size.insert(mode_tensor_size.end(), tensor_size.begin(),
                                                    tensor_size.begin()+mode_k);
    mode_tensor_size.insert(mode_tensor_size.end(), tensor_size.begin()+(mode_k+1),
                                                    tensor_size.end());

    std::vector<size_t> mode_tensor_index = TensorIndex(mode_flat_index, mode_tensor_size);
    mode_tensor_index.insert(mode_tensor_index.begin()+mode_k, 0);
    return mode_tensor_index;
}


/**
 *
 * The paper uses RowMajor (aka Lexicographic) as opposed to ColMajor (aka Colexicographic)
 */
// I'm hoping any call to these will be inlined by the compiler, 
// but I don't know if they will, will test


float& Tensor::operator[](const std::vector<size_t>& tensor_index) {

    size_t flat_index = FlatIndex(tensor_index, tensor_size);
    #ifdef DEBUG
    if(flat_index >= NumComponents()) {
        std::stringstream ss; 
        ss << tensor_index;
        std::cerr << "Error Tensor::operator[]: flat_index >= NumComponents() "
                     " flat_index == " << flat_index << " tensor_index == " << ss.str() << "\n";
    }
    #endif // DEBUG
    return data[flat_index];
}

const float& Tensor::operator[](const std::vector<size_t>& tensor_index) const {  

    size_t flat_index = FlatIndex(tensor_index, tensor_size);
    #ifdef DEBUG
    if(flat_index >= NumComponents()) {
        std::stringstream ss; 
        ss << tensor_index;
        std::cerr << "Error Tensor::operator[]: flat_index >= NumComponents() "
                     " flat_index == " << flat_index << " tensor_index == " << ss.str() << "\n";
    }
    #endif // DEBUG
    return data[flat_index];
}


float& Tensor::operator[](size_t flat_index) {
    #ifdef DEBUG
    if(flat_index >= NumComponents()) {
        std::cerr << "Error Tensor::operator[]: flat_index >= NumComponents()\n";
    }
    #endif // DEBUG
   
    return data[flat_index];
}



const float& Tensor::operator[](size_t flat_index) const {
    #ifdef DEBUG
    if(flat_index >= NumComponents()) {
        std::cerr << "Error Tensor::operator[]: flat_index >= NumComponents()\n";
    }
    #endif // DEBUG
   
    return data[flat_index];

}




float* Tensor::Data() {
    return data;
}



void Tensor::Resize(std::vector<size_t> tensor_size_in) {
    #ifdef DEBUG
    if(MultiplyElements(tensor_size_in) != num_components) {
        std::cerr 
        << "Error Tensor::Resize: MultiplyElements(tensor_size_in) != num_components\n";
    }
    #endif // DEBUG

    tensor_size = std::move(tensor_size_in); 
}

void Tensor::SetZero() {
    for(size_t i = 0; i < num_components; i++) {
        data[i] = 0;
    }
}

void Tensor::SetUniformRandom(float lower_bound, float upper_bound) {

    std::random_device rd;
    std::mt19937 gen;

    static std::uniform_real_distribution<float> dist(lower_bound, upper_bound); 
	for(size_t i = 0, N = NumComponents(); i < N; i++) {
        data[i] = dist(gen);
    }
}


void Tensor::SwapAxes(size_t mode_k, size_t mode_l) { 
    #ifdef DEBUG
    if(mode_k >= Order() || mode_l >= Order()) {
        std::cerr << "Error Tensor::SwapAxes: mode_k >= Order() || mode_l >= Order()\n";
    }
    #endif // DEBUG

    // there's probably a better algorithm
    // \todo this is broken if I plan to use this class for operating on borrowed data
    //       from a pytorch tensor
    Tensor swapped = SwappedAxes(mode_k, mode_l);
    *this = swapped;
}

Tensor Tensor::SwappedAxes(size_t mode_k, size_t mode_l) const { 
    #ifdef DEBUG
    if(mode_k >= Order() || mode_l >= Order()) {
        std::cerr << "Error Tensor::SwapAxes: mode_k >= Order() || mode_l >= Order()\n";
    }
    #endif // DEBUG

    std::vector<size_t> swapped_tensor_size = tensor_size;
    {
    size_t temp = swapped_tensor_size[mode_k];
    swapped_tensor_size[mode_k] = swapped_tensor_size[mode_l];
    swapped_tensor_size[mode_l] = temp;
    }


    Tensor swapped(swapped_tensor_size);

    for(size_t i = 0; i < num_components; i++) {
        std::vector<size_t> ti = TensorIndex(i, tensor_size);
        std::vector<size_t> swapped_ti = ti;
        {
        size_t temp = swapped_ti[mode_k];
        swapped_ti[mode_k] = swapped_ti[mode_l];
        swapped_ti[mode_l] = temp;
        }

        size_t swapped_i = FlatIndex(swapped_ti, swapped_tensor_size);

        swapped.data[swapped_i] = data[i];
    }

    return swapped;
}




bool SameTensorSize(const Tensor& x, const Tensor& y) {
    if(x.num_components != y.num_components || x.tensor_size.size() != y.tensor_size.size()) {
        return false;
    }

    for(size_t i = 0; i < x.tensor_size.size(); i++) {
        if(x.tensor_size[i] != y.tensor_size[i]) {
            return false;
        }
    }

    return true;
}

Tensor Contract(size_t mode_k, size_t mode_l, const Tensor& x, const Tensor& y) {
    // \todo what is the definition of contracting a tensor of order 0 with another tensor?
    #ifdef DEBUG
    if(x.Order() <= mode_k || y.Order() <= mode_l) {
        std::cerr << "Error Tensor::Contract x.Order() <= mode_k || y.Order() <= mode_l\n";
    }
    if(x.tensor_size[mode_k] != y.tensor_size[mode_l]) {
        std::cerr << "Error Tensor::Contract: x.tensor_size[mode_k] != y.tensor_size[mode_l]\n"; 
    } 
    #endif // DEBUG

    size_t x_ord = x.Order();
    size_t y_ord = y.Order();

    // for some reason I do this task with the stl methods later down
    
    std::vector<size_t> contraction_size; contraction_size.reserve(x_ord + y_ord - 2); 
    for(size_t i = 0; i < x_ord; i++) {
        if(i != mode_k) {
            contraction_size.push_back(x.tensor_size[i]);
        }
    }
    for(size_t i = 0; i < y_ord; i++) {
        if(i != mode_l) {
            contraction_size.push_back(y.tensor_size[i]);
        }
    }

    Tensor contraction(contraction_size);  
    float* c_data = contraction.data;
    size_t c_data_size =  contraction.NumComponents();
    size_t mode_size = x.tensor_size[mode_k]; // == y.tensor_size[mode_l]

    size_t x_offset = ModeStride(mode_k, x.tensor_size);
    size_t y_offset = ModeStride(mode_l, y.tensor_size);
    
    for(size_t i = 0; i < c_data_size; i++) {
        /**
         * this i is the flat index corresponding to the tensor index
         * (i0, i1, ..., i(k-1), i(k+1), ..., in,
         *  j0, j1, ..., j(l-1), j(l+1), ..., jm) (using notation from the paper)
         */

        // \todo converting between flat index and tensor index like this in a tight loop is
        // naive and slow, and just an easy first try... when I change to computing the
        // x_initial/y_initial without first converting to TensorIndex, I'll have to
        // have two separate methods RowMajorContract and ColMajorContract
        std::vector<size_t> c_tensor_index = TensorIndex(i, contraction_size);
        auto c_it = c_tensor_index.begin();
        
        // compute x_initial, the first index of the given mode vector in the x tensor
        std::vector<size_t> x_initial_ti; x_initial_ti.reserve(x_ord); // "x_initial_tensor_index"
        x_initial_ti.insert(x_initial_ti.end(), c_it, c_it+mode_k);
        x_initial_ti.push_back(0);
        x_initial_ti.insert(x_initial_ti.end(), c_it+mode_k, c_it+(x_ord-1));
        /**
         * 0 <= x_initial < x_ord/x.tensor_size[mode_k]
         * x_initial is the flat index corresponding to the tensor index
         * (i0, i1, ..., i(k-1), 0, i(k+1), ..., in) (using notation from the paper)
         */
        size_t x_initial = FlatIndex(x_initial_ti, x.tensor_size); 
                
        // compute y_initial
        std::vector<size_t> y_initial_ti; y_initial_ti.reserve(y_ord); // "y_initial_tensor_index"
        y_initial_ti.insert(y_initial_ti.end(), c_it+(x_ord-1), c_it+(x_ord-1+mode_l));
        y_initial_ti.push_back(0);
        y_initial_ti.insert(y_initial_ti.end(), 
                            c_it+(x_ord-1+mode_l), c_it+(c_tensor_index.size()));
        size_t y_initial = FlatIndex(y_initial_ti, y.tensor_size);
        
        float dot_product = 0; 
        for(size_t j = 0; j < mode_size; j++) {
            dot_product += x.data[x_initial + j*x_offset] * y.data[y_initial + j*y_offset]; 
        }
        c_data[i] = dot_product;
    }

    return contraction;
} // Contract

Tensor MultiplyMatrix(size_t mode_k, const Tensor& t, const Tensor& m) {
    #ifdef DEBUG
    if(m.Order() != 2) {
        std::cerr << "Error MultiplyMatrix: matrix.Order() != 2\n";
    }
    if(t.tensor_size[mode_k] != m.tensor_size[0]) {
        std::cerr << "Error Tensor::MultiplyMatrix: t.tensor_size[mode_k] != m.tensor_size[0]\n"; 
    }
    #endif // DEBUG

    std::vector<size_t> out_size = t.tensor_size; 
    out_size[mode_k] = m.tensor_size[1];

    Tensor out(out_size);
    float* o_data = out.data;
    size_t o_data_size = out.NumComponents();
    size_t mode_size = t.tensor_size[mode_k];

    size_t t_offset = ModeStride(mode_k, t.tensor_size);
    size_t m_offset = ModeStride(0, m.tensor_size);

    for(size_t i = 0; i < o_data_size; i++) {
        std::vector<size_t> t_initial_ti = TensorIndex(i, out_size);
        
        std::vector<size_t> m_initial_ti = {0, t_initial_ti[mode_k]};
        size_t m_initial = FlatIndex(m_initial_ti, m.tensor_size);

        t_initial_ti[mode_k] = 0;
        size_t t_initial = FlatIndex(t_initial_ti, t.tensor_size);

        float dot_product = 0;
        for(size_t j = 0; j < mode_size; j++) {
            dot_product += t.data[t_initial + j*t_offset] * m.data[m_initial + j*m_offset];
        }
        o_data[i] = dot_product;
    }

    return out;
} // MultiplyMatrix



Tensor Convolve(size_t mode_k, size_t mode_l, const Tensor& x, const Tensor& y) {
     // \todo what is the definition of convolving a tensor of order 0 with another tensor?
    #ifdef DEBUG
    if(x.Order() <= mode_k || y.Order() <= mode_l) {
        std::cerr << "Error Tensor::Convolve x.Order() <= mode_k || y.Order() <= mode_l\n";
    } 
    #endif // DEBUG
 
    size_t x_ord = x.Order();
    size_t y_ord = y.Order();

    std::vector<size_t> convolution_size; convolution_size.reserve(x_ord + y_ord - 1); 
    size_t x_vector_length = x.tensor_size[mode_k];
    size_t y_vector_length = y.tensor_size[mode_l];
    size_t conv_vector_length = x_vector_length + y_vector_length - 1;
    for(size_t i = 0; i < x_ord; i++) {
        if(i != mode_k) {
            convolution_size.push_back(x.tensor_size[i]);
        } else {
            convolution_size.push_back(conv_vector_length);
        }
    }
    for(size_t i = 0; i < y_ord; i++) {
        if(i != mode_l) {
            convolution_size.push_back(y.tensor_size[i]);
        }
    } 

    std::vector<size_t> conv_slice_size; conv_slice_size.reserve(x_ord+y_ord-2);
    conv_slice_size.insert(conv_slice_size.end(), 
                           convolution_size.begin(), convolution_size.begin()+mode_k);
    conv_slice_size.insert(conv_slice_size.end(), // skip the mode_k entry
                           convolution_size.begin()+mode_k+1, convolution_size.end());

    Tensor convolution(convolution_size);  
    float* c_data = convolution.data;
    size_t c_slice_data_size = convolution.NumComponents()/conv_vector_length;

    size_t x_offset = ModeStride(mode_k, x.tensor_size);
    size_t y_offset = ModeStride(mode_l, y.tensor_size);
    size_t c_offset = ModeStride(mode_k, convolution_size);

    for(size_t i = 0; i < c_slice_data_size; i++) {
        /**
         * this i is the flat index corresponding to the tensor slice index
         * (i0, i1, ..., i(k-1), :, i(k+1), ..., in,
         *  j0, j1, ..., j(l-1), j(l+1), ..., jm) (using notation from the paper)
         */

        // \todo optimize out these conversions between FlatIndex and TensorIndex
        
        std::vector<size_t> c_tensor_index = TensorIndex(i, conv_slice_size); 
        auto c_it = c_tensor_index.begin();

        // compute x_initial, the first index of the given mode vector in the x tensor
        std::vector<size_t> x_initial_ti; x_initial_ti.reserve(x_ord);// "x_initial_tensor_index"
        x_initial_ti.insert(x_initial_ti.end(), c_it, c_it+mode_k); 
        x_initial_ti.push_back(0);
        x_initial_ti.insert(x_initial_ti.end(), c_it+mode_k, c_it+c_tensor_index.size());
        size_t x_initial = FlatIndex(x_initial_ti, x.tensor_size); 

        // compute y_initial
        std::vector<size_t> y_initial_ti; y_initial_ti.reserve(y_ord);// "y_initial_tensor_index"
        y_initial_ti.insert(y_initial_ti.end(), c_it+(x_ord), c_it+(x_ord+mode_l));
        y_initial_ti.push_back(0);
        y_initial_ti.insert(y_initial_ti.end(), 
                            c_it+(x_ord+mode_l), c_it+(c_tensor_index.size()));
        size_t y_initial = FlatIndex(y_initial_ti, y.tensor_size);

        // see https://www.mathworks.com/help/matlab/ref/conv.html
        for(size_t k = 0; k < conv_vector_length; k++) {
            float conv_sum = 0;
            for(int j = 0; int(k)-j >= 0; j++) { 
                float xp = j < int(x_vector_length) ? x.data[x_initial+j*x_offset] : 0;
                float yp 
                 = int(k)-j < int(y_vector_length) ? y.data[y_initial+(int(k)-j)*y_offset] : 0;
                conv_sum += xp*yp;
            }
            
            c_data[i + k*c_offset] = conv_sum;
        }
    }

    return convolution;
} // Convolve


Tensor PartialOuterProduct(size_t mode_k, size_t mode_l, 
                           const Tensor& x, const Tensor& y)
{
    // \todo what is the definition of partial outer product with a tensor of order 0
    #ifdef DEBUG
    if(x.Order() <= mode_k || y.Order() <= mode_l) {
        std::cerr << "Error Tensor::PartialOuterProduct: "
                     "x.Order() <= mode_k || y.Order() <= mode_l\n";
    }
    if(x.tensor_size[mode_k] != y.tensor_size[mode_l]) {
        std::cerr << "Error Tensor::PartialOuterProduct: "
                     "x.tensor_size[mode_k] != y.tensor_size[mode_l]\n"; 
    }
    #endif // DEBUG

 
    size_t x_ord = x.Order();
    size_t y_ord = y.Order();

    std::vector<size_t> partial_size; partial_size.reserve(x_ord + y_ord - 1); 
         
    partial_size.insert(partial_size.end(), x.tensor_size.begin(), x.tensor_size.end()); 
    partial_size.insert(partial_size.end(), 
                        y.tensor_size.begin(), y.tensor_size.begin()+mode_l);
    partial_size.insert(partial_size.end(), 
                        y.tensor_size.begin()+(mode_l+1), y.tensor_size.end());

    Tensor partial(partial_size);

    size_t p_data_size = partial.NumComponents();

    for(size_t i = 0; i < p_data_size; i++) {
         std::vector<size_t> p_tensor_index = TensorIndex(i, partial_size);
        auto p_it = p_tensor_index.begin();

        std::vector<size_t> x_tensor_index; x_tensor_index.reserve(x_ord);
        x_tensor_index.insert(x_tensor_index.end(), p_it, p_it+x_ord);
        size_t x_index = FlatIndex(x_tensor_index, x.tensor_size);

        std::vector<size_t> y_tensor_index; y_tensor_index.reserve(y_ord);
        y_tensor_index.insert(y_tensor_index.end(), p_it+x_ord, p_it+(x_ord+mode_l));
        y_tensor_index.push_back(x_tensor_index[mode_k]);
        y_tensor_index.insert(y_tensor_index.end(), p_it+(x_ord+mode_l+1), p_tensor_index.end());
        size_t y_index = FlatIndex(y_tensor_index, y.tensor_size);

        partial.data[i] = x.data[x_index]*y.data[y_index];
       
    }

    return partial;
} // PartialOuterProduct


Tensor OuterProduct(const Tensor& x, const Tensor& y) {

    size_t x_ord = x.Order();
    size_t y_ord = y.Order();

    std::vector<size_t> product_size; product_size.reserve(x_ord + y_ord);
    product_size.insert(product_size.end(), x.tensor_size.begin(), x.tensor_size.end());
    product_size.insert(product_size.end(), y.tensor_size.begin(), y.tensor_size.end());
    Tensor product(product_size); 
    
    size_t p_data_size = product.NumComponents();

    // \todo there's probably a faster method than converting between FlatIndex
    // and TensorIndex
    for(size_t i = 0; i < p_data_size; i++) {
        std::vector<size_t> p_tensor_index = TensorIndex(i, product_size);
        auto p_it = p_tensor_index.begin();

        std::vector<size_t> x_tensor_index; x_tensor_index.reserve(x_ord);
        x_tensor_index.insert(x_tensor_index.end(), p_it, p_it+x_ord);
        size_t x_index = FlatIndex(x_tensor_index, x.tensor_size);

        std::vector<size_t> y_tensor_index; y_tensor_index.reserve(y_ord);
        y_tensor_index.insert(y_tensor_index.end(), p_it+x_ord, p_it+(x_ord+y_ord));
        size_t y_index = FlatIndex(y_tensor_index, y.tensor_size);

        product.data[i] = x.data[x_index]*y.data[y_index];
    }

    return product;
} // OuterProduct


float InnerProduct(const Tensor& x, const Tensor& y) {

    #ifdef DEBUG
    if(!SameTensorSize(x, y)) {
        std::cerr << "Error InnerProduct: !SameTensorSize(x, y)\n";
    }
    #endif // DEBUG

    float sum = 0;
    for(size_t i = 0, N = x.num_components; i < N; i++) {
        sum += x.data[i] * y.data[i];
    }

    return sum;
}

float Norm(const Tensor& x) {
    return std::sqrt(InnerProduct(x, x));
}


/*
CPDDecompOut CPDDecompALSOrder3(const Tensor& X, size_t rank) {
    #ifdef DEBUG
    if(X.Order() != 3) {
        std::cerr << "Error CPDDecompALSOrder3: X.Order() != 3\n";
    }
    #endif // DEBUG

    Tensor A({X.tensor_size[0], X.tensor_size[1] rank}); // factor matrices

	A.SetUniformRandom(0, 1);

	size_t num_iterations = 5;

}
*/



size_t Tensor::Order() const {
    return tensor_size.size(); 
}

size_t Tensor::NumComponents() const { 
    return num_components; 
}

std::vector<size_t> Tensor::TensorSize() const {
    return tensor_size;
}

std::string Tensor::FlatString() const {
    std::stringstream ss;
     
    size_t N = NumComponents();
    
    if(N == 0) {
        return "";   
    }
    for(size_t i = 0; i < N; i++) {
        ss << data[i];
        if(i < N-1) {
            ss << " ";
        }
    }
    return ss.str();
}



} // OPS
