#include <iostream>

#include "Profiling.h"
#include "HelpersTypedefs.h"
#include "Tensor.h"

#define DEBUG



OPS::Prof ops_prof; // global profiling object
int main() {
    using namespace OPS;

    /* 
    std::vector<size_t> tensor_index = {1, 0};
    std::vector<size_t> tensor_size = {2, 2}; 
    size_t flat_index = FlatIndex(tensor_index, tensor_size);

    std::cout << "tensor_size " << tensor_size << "\n";
    std::cout << "tensor_index " << tensor_index << "\n";
    std::cout << "flat_index from tensor_index " << flat_index << "\n";
    std::cout << "tensor_index from flat_index " << TensorIndex(flat_index, tensor_size) << "\n";
    */
 
    
    Tensor t1({3,2});
    Tensor t2({2,3});

    t1[{0,0}] = 1; t1[{0,1}] = 0;
    t1[{1,0}] = 0; t1[{1,1}] = 1;
    t1[{2,0}] = 5; t1[{2,1}] = 1;

    t2[{0,0}] = 1; t2[{0,1}] = 0; t2[{0,2}] = 1;
    t2[{1,0}] = 0; t2[{1,1}] = 1; t2[{1,2}] = 1;
    
   
    Tensor t3; 
    for(size_t i = 0; i < 50; i++) {
    ops_prof.Mark(0);
    t3 = Contract(1, 0, t1, t2); 
    ops_prof.Mark(0, "Contract", true);
    }
    ops_prof.PrintInfo(0);
 
    std::cout << t3.FlatString() << "\n";
  

  /*  
    Tensor t1({3, 2, 2});
    Tensor t2({2, 3, 2});

    t1[{0,0,0}] = 1; t1[{0,0,1}] = 0;
    t1[{0,1,0}] = 0; t1[{0,1,1}] = 1;
    
    t1[{1,0,0}] = 1; t1[{1,0,1}] = 0;
    t1[{1,1,0}] = 0; t1[{1,1,1}] = 1;
    
    t1[{2,0,0}] = 1; t1[{2,0,1}] = 0;
    t1[{2,1,0}] = 0; t1[{2,1,1}] = 1;
    
    
    t2[{0,0,0}] = 1; t2[{0,0,1}] = 0;
    t2[{0,1,0}] = 0; t2[{0,1,1}] = 1;
    t2[{0,2,0}] = 0; t2[{0,2,1}] = 0;
 
    t2[{1,0,0}] = 1; t2[{1,0,1}] = 0;
    t2[{1,1,0}] = 0; t2[{1,1,1}] = 1;
    t2[{1,2,0}] = 0; t2[{1,2,1}] = 0;

    Tensor t3 = Contract(2, 0, t1, t2);
    std::cout << t3.Order() << "\n";
    std::cout << t3.FlatString() << "\n";
   */

    /*  
    Tensor t1({2,2});
    t1[{0,0}] = 1; t1[{0,1}] = 1;  
    t1[{1,0}] = 1; t1[{1,1}] = 0;

    Tensor t2({4});
    t2[0] = 1; t2[1] = 1; t2[2] = 1; t2[3] = 2;

    Tensor t3 = Convolve(0, 0, t1, t2);
    std::cout << t3.Order() << "\n";
    std::cout << t3.FlatString() << "\n";
    */

    /*  
    Tensor t1({2});
    t1[0] = 1; t1[1] = 1;

    Tensor t2({4});
    t2[0] = 1; t2[1] = 1; t2[2] = 1; t2[3] = 1;
    

    Tensor t3 = OuterProduct(t1, t2);
    std::cout << t3.Order() << "\n";
    std::cout << t3.FlatString() << "\n";
    */

/*
    Tensor t1({2,2});
    t1[0] = 1; t1[1] = 1; t1[2] = 1; t1[3] = 1;

    Tensor t2({2});
    t2[0] = 1; t2[1] = 0; 
    

    Tensor t3 = PartialOuterProduct(0, 0, t1, t2);
    std::cout << t3.Order() << "\n";
    std::cout << t3.FlatString() << "\n";
*/


    return 0;
}

