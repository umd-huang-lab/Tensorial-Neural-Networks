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
 
    /*    
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
    */

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
/*
    Tensor t1({2,2,2});
    t1[{0,0,0}] = 1; t1[{0,1,0}] = 1;  
    t1[{1,0,0}] = 1; t1[{1,1,0}] = 0;
    t1[{0,0,1}] = 1; t1[{0,1,1}] = 1;  
    t1[{1,0,1}] = 1; t1[{1,1,1}] = 0;

    Tensor t2({2,2,2});
    t2[{0,0,0}] = 1; t2[{0,1,0}] = 1;  
    t2[{1,0,0}] = 1; t2[{1,1,0}] = 0;
    t2[{0,0,1}] = 1; t2[{0,1,1}] = 0;  
    t2[{1,0,1}] = 1; t2[{1,1,1}] = 0;
    std::cout << "InnerProduct = " << InnerProduct(t1, t2) << "\n";
*/

    /**
     * Rank decomposition section 3.1 of Kolda
     */

    Tensor X({2,2,2});
    X[{0,0,0}] = 1; X[{0,1,0}] = 0;
    X[{1,0,0}] = 0; X[{1,1,0}] = 1;

    X[{0,0,1}] = 0; X[{0,1,1}] = 1;
    X[{1,0,1}] =-1; X[{1,1,1}] = 0;

    Tensor A1({2});
    A1[0] = 1; A1[1] = 0;

    Tensor A2({2});
    A2[0] = 0; A2[1] = 1;

    Tensor A3({2});
    A3[0] = 1; A3[1] =-1;


    Tensor B1({2});
    B1[0] = 1; B1[1] = 0;

    Tensor B2({2});
    B2[0] = 0; B2[1] = 1;

    Tensor B3({2});
    B3[0] = 1; B3[1] = 1;


    Tensor C1({2});
    C1[0] = 1; C1[1] =-1;

    Tensor C2({2});
    C2[0] = 1; C2[1] = 1;

    Tensor C3({2});
    C3[0] = 0; C3[1] = 1;


    Tensor decomp = OuterProduct(OuterProduct(A1, B1), C1)
                  + OuterProduct(OuterProduct(A2, B2), C2)  
                  + OuterProduct(OuterProduct(A3, B3), C3);
    
    std::cout << X.FlatString() << "\n";
    std::cout << decomp.FlatString() << "\n";

/*
    Tensor o = OuterProduct(OuterProduct(A1, B1), C1) + OuterProduct(OuterProduct(A1, B1), C1);
    std::cout << o.FlatString() << "\n";
    Tensor c = o;
    std::cout << c.FlatString() << "\n";
    Tensor d = c;
    std::cout << d.FlatString() << "\n";
    //std::cout << decomp.TensorSize() << "\n";
*/


    return 0;
}

