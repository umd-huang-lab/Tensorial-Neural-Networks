#include <iostream>

#include "Profiling.h"
#include "HelpersTypedefs.h"
#include "Tensor.h"
#include "TensorNetwork.h"

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
    /*
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
    std::cout << (X - decomp).FlatString() << "\n";
    std::cout << Norm(X - decomp) << "\n"; 
    */

/*
    Tensor o = OuterProduct(OuterProduct(A1, B1), C1) + OuterProduct(OuterProduct(A1, B1), C1);
    std::cout << o.FlatString() << "\n";
    Tensor c = o;
    std::cout << c.FlatString() << "\n";
    Tensor d = c;
    std::cout << d.FlatString() << "\n";
    //std::cout << decomp.TensorSize() << "\n";
*/

/*
    Tensor X({2,2,2});
    X[{0,0,0}] = 1; X[{0,1,0}] = 0;
    X[{1,0,0}] = 0; X[{1,1,0}] = 1;

    X[{0,0,1}] = 0; X[{0,1,1}] = 1;
    X[{1,0,1}] =-1; X[{1,1,1}] = 0;

    Tensor M({2,3});
    M[{0,0}] = 1; M[{0,1}] = 0; M[{0,2}] = 0;
    M[{1,0}] = 0; M[{1,1}] = 1; M[{1,2}] = 0;

    Tensor out = MultiplyMatrix(2, X, M);
    std::cout << out.TensorSize() << "\n";
    std::cout << out.FlatString() << "\n";
*/

/*
    Tensor X({2,2,3});
    X[{0,0,0}] = 1; X[{0,1,0}] = 0;
    X[{1,0,0}] = 0; X[{1,1,0}] = 1;

    X[{0,0,1}] = 0; X[{0,1,1}] = 1;
    X[{1,0,1}] =-1; X[{1,1,1}] = 0;

    X[{0,0,2}] = 0; X[{0,1,2}] = 1;
    X[{1,0,2}] =-1; X[{1,1,2}] = 0;

    std::cout << "Mode Tensor Index: " << ModeTensorIndex(1, 3,  X.TensorSize()) << "\n";  
*/


/*
    {
    Tensor X({2,2,3});
    X[{0,0,0}] = 1; X[{0,1,0}] = 2;
    X[{1,0,0}] = 0; X[{1,1,0}] = 1;

    X[{0,0,1}] = 0; X[{0,1,1}] = 1;
    X[{1,0,1}] =-1; X[{1,1,1}] = 3;

    X[{0,0,2}] =-5; X[{0,1,2}] = 1;
    X[{1,0,2}] =-1; X[{1,1,2}] = 5;

    // In COL_MAJOR X is 1 0 2 1 0 -1 1 3 -5 -1 1 5
    // In ROW_MAJOR X is 1 0 -5 2 1 1 0 -1 -1 1 3 5

    std::cout << X.FlatString() << "\n";
    X.SwapAxes(0,1);
    std::cout << X.FlatString() << "\n";
    }
    {
    Tensor X({2,2});
    X[{0,0}] = 1; X[{0,1}] = 2;
    X[{1,0}] = 0; X[{1,1}] = 1;


    std::cout << X.FlatString() << "\n";
    X.SwapAxes(0,1);
    std::cout << X.FlatString() << "\n";
    }
*/

/*
    {
    // Kolda writes on page 461 that MultiplyMatrix is independent of the 
    // multiplication order for different modes

    Tensor X({2,2,3});
    X[{0,0,0}] = 1; X[{0,1,0}] = 2;
    X[{1,0,0}] = 0; X[{1,1,0}] = 1;

    X[{0,0,1}] = 0; X[{0,1,1}] = 1;
    X[{1,0,1}] =-1; X[{1,1,1}] = 3;

    X[{0,0,2}] =-5; X[{0,1,2}] = 1;
    X[{1,0,2}] =-1; X[{1,1,2}] = 5;
    
    Tensor A({2,2});
    A[{0,0}] = 2; A[{0,1}] = 0;
    A[{1,0}] = 0; A[{1,1}] = 2;

    Tensor B({3,2});
    B[{0,0}] = 2; B[{0,1}] = 0;
    B[{1,0}] = 0; B[{1,1}] = 2;
    B[{2,0}] = 2; B[{2,1}] = 2;
    {
    Tensor t1 = MultiplyMatrix(1, X, A);
    t1 = MultiplyMatrix(2, t1, B);

    Tensor t2 = MultiplyMatrix(2, X, B);
    t2 = MultiplyMatrix(1, t2, A); 
    std::cout << "t1: " << t1.FlatString() << "\n";
    std::cout << "t2: " << t2.FlatString() << "\n";
    std::cout << "dist(t1, t2) = " << (Norm(t1 - t2)) << "\n";
    }
    {
    // Note that Kolda has different conventions for multiply matrix than us...
    // The difference implies a difference in the identity for same mode subsequent 
    // matrix mulpliplies 
    // Hers says X x_n B x_n A = X x_n (AB)
    // whereas ours says X x_n B x_n A = X x_n (BA) (associativity of the action)
    // 
    Tensor t1 = MultiplyMatrix(2, X, B);
    t1 = MultiplyMatrix(2, t1, A);

    Tensor BA = Contract(1, 0, B, A);
    Tensor t2 = MultiplyMatrix(2, X, BA);

    std::cout << "t1: " << t1.FlatString() << "\n";
    std::cout << "t2: " << t2.FlatString() << "\n";
    std::cout << "dist(t1, t2) = " << (Norm(t1 - t2)) << "\n";
    }
    {
    // confirming MultiplyMatrix agrees with normal matrix multiplication
     
    Tensor BA1 = Contract(1, 0, B, A);
    Tensor BA2 = MultiplyMatrix(1, B, A);
    std::cout << "BA1: " << BA1.FlatString() << "\n";
    std::cout << "BA2: " << BA2.FlatString() << "\n";
    std::cout << "dist(BA1, BA2) = " << (Norm(BA1 - BA2)) << "\n";
    }
    }
*/
/*
    {
    Tensor M({2,3});
    M[{0,0}] = 12; M[{0,1}] = 108; M[{0,2}] = 12;
    M[{1,0}] = 31; M[{1,1}] = 100; M[{1,2}] = 35;

    Tensor S({3,2});
    S[{0,0}] = M[{0,0}]; S[{1,0}] = M[{0,1}]; S[{2,0}] = M[{0,2}];
    S[{0,1}] = M[{1,0}]; S[{1,1}] = M[{1,1}]; S[{2,1}] = M[{1,2}];

    std::cout << "Norm(M^T - S) = " << Norm(M.SwappedAxes(0,1) - S) << "\n";
    }
*/


/*
    { 
    Tensor M({3,3});
    M[{0,0}] = 9; M[{0,1}] = 10; M[{0,2}] = 1022;
    M[{1,0}] = 7; M[{1,1}] = 41; M[{1,2}] =-10930;
    M[{2,0}] = 1209; M[{2,1}] = -8; M[{2,2}] = 10954;
    
    // 
    //Tensor M({2,2});
    //M[{0,0}] = 1; M[{0,1}] = 1; 
    //M[{1,0}] = 0; M[{1,1}] = 1;
    //

    std::cout << "M: " << M.FlatString() << " with tensor size: " << M.TensorSize() << "\n";

    Tensor M_copy = M;
    M_copy.SetPseudoInverseTranspose(0, 1, {0,0});

    std::cout << "M Pseudo Inverse Transpose: " << M_copy.FlatString() << " with tensor size: " << M_copy.TensorSize() << "\n";

    Tensor P = M_copy.SwappedAxes(0, 1);
    std::cout << "M Pseudo Inverse: " << P.FlatString() << " with tensor size: " << P.TensorSize() << "\n";

    Tensor P2 = M.CalcPseudoInverse(0, 1, {0,0});
    std::cout << "M Pseudo Inverse by CalcPseudoInverse: " << P2.FlatString() 
              << " with tensor size: " << P2.TensorSize() << "\n";
    std::cout << "Norm(P - P2): " << Norm(P - P2) << "\n";

    Tensor I = Contract(1, 0, M, P);
    Tensor I2 = Contract(1, 0, P, M);
    std::cout << "M * P = " << I.FlatString() << "\n";
    std::cout << "P * M = " << I2.FlatString() << "\n";
    std::cout << "Norm(P*M - M*P): " << Norm(I - I2) << "\n";

    }
*/


/*
    {
    TensorNetworkDefinition network;
    network.AddNode("A", 2);
    network.AddNode("B", 1);
    network.AddNode("C", 1);
 
    network.AddEdge("j", {{"A", 1}}, Operation::CONTRACTION, 0);
    network.AddEdge("i", {{"A", 0}, {"B", 0}, {"C", 0}}); 
    

    Tensor A({2,2});
    A[0] = 1; A[1] = 1; A[2] = 1; A[3] = 1; 
    
    Tensor B({2});
    B[0] = 1; B[1] = 1; 

    Tensor C({2});
    C[0] = 1; C[1] = 1; 

    Tensor out = network.Evaluate({std::move(A), std::move(B), std::move(C)});

    std::cout << "out order: " << out.Order() << "\n";
    std::cout << "out tensor_size: " << out.TensorSize() << "\n";
    std::cout << "out tensor_size.size(): " << out.TensorSize().size() << "\n";

    std::cout << "out " << out.FlatString() << "\n";

    }
*/
/*
    {
    TensorNetworkDefinition network;
    network.AddNode("A", 2);
    network.AddEdge("i", {{"A", 0}});

    Tensor A({2,1});
    A[{0,0}] = 1; A[{1,0}] = 2;

    Tensor out = network.Evaluate({std::move(A)});

    std::cout << "out order: " << out.Order() << "\n";
    std::cout << "out tensor_size: " << out.TensorSize() << "\n";
    std::cout << "out tensor_size.size(): " << out.TensorSize().size() << "\n";

    std::cout << "out " << out.FlatString() << "\n";

    }
*/
/*
    {
    TensorNetworkDefinition network;
    network.AddNode("A", 1);
    network.AddNode("B", 1);
    network.AddEdge("i", {{"A", 0}, {"B", 0}}, Operation::CONVOLUTION, 0); 

    Tensor A({2});
    A[0] = 1; A[1] = 2;

    Tensor B({2});
    B[0] = 3; B[1] = 4;

    Tensor out = network.Evaluate({std::move(A), std::move(B)});

    std::cout << "out order: " << out.Order() << "\n";
    std::cout << "out tensor_size: " << out.TensorSize() << "\n";
    std::cout << "out tensor_size.size(): " << out.TensorSize().size() << "\n";

    std::cout << "out " << out.FlatString() << "\n";


    }
*/
/*
    {
    TensorNetworkDefinition network;
    network.AddNode("A", 1);
    network.AddNode("B", 2);
    network.AddNode("C", 1);
    network.AddNode("D", 1);
    network.AddEdge("i", {{"A", 0}, {"B", 0}, {"C", 0}}, Operation::CONVOLUTION, 0); 
    network.AddEdge("j", {{"D", 0}});

    Tensor A({2});
    A[0] = 1; A[1] = 2;

    Tensor B({2, 2});
    B[{0,0}] = 3; B[{1,0}] = 4;
    B[{0,1}] = 5; B[{1,1}] = 6;

    Tensor C({2});
    C[0] = 1; C[1] = 3;

    Tensor D({2});
    D[0] = 2; D[1] = 3;

    Tensor out = network.Evaluate({A, B, C, D});

    std::cout << "out order: " << out.Order() << "\n";
    std::cout << "out tensor_size: " << out.TensorSize() << "\n";
    std::cout << "out tensor_size.size(): " << out.TensorSize().size() << "\n";

    std::cout << "out " << out.FlatString() << "\n";

    }
*/



    {

    TensorNetworkDefinition network;
    network.SetConvolutionType(ConvolutionType::SAME);
    network.AddNode("A", 1);
    network.AddNode("B", 1);
    network.AddEdge("i", {{"A", 0}, {"B", 0}}, Operation::CONVOLUTION, 0);   

    Tensor A({3}); 
    A[0] = 1; A[1] = 2; A[2] = 3;

    Tensor B({2});
    B[0] = 3; B[1] = 4;

    Tensor out = network.Evaluate({std::move(A), std::move(B)});

    std::cout << "out order: " << out.Order() << "\n";
    std::cout << "out tensor_size: " << out.TensorSize() << "\n";
    std::cout << "out tensor_size.size(): " << out.TensorSize().size() << "\n";

    std::cout << out.FlatString() << "\n";
    }


/*
    {

    TensorNetworkDefinition network;
    network.SetConvolutionType(ConvolutionType::SAME);
    network.AddNode("A", 1);
    network.AddNode("B", 2);
    network.AddEdge("i", {{"A", 0}, {"B", 0}}, Operation::CONVOLUTION, 0);  
    network.AddEdge("j", {{"B", 1}}, Operation::CONTRACTION); 

    Tensor A({2}); 
    A[0] = 1; A[1] = 2;

    Tensor B({2,1});
    B[{0,0}] = 3; B[{1,0}] = 4;

    Tensor out = network.Evaluate({std::move(A), std::move(B)});

    std::cout << "out order: " << out.Order() << "\n";
    std::cout << "out tensor_size: " << out.TensorSize() << "\n";
    std::cout << "out tensor_size.size(): " << out.TensorSize().size() << "\n";

    std::cout << out.FlatString() << "\n";
    }
*/
/*
    {

    TensorNetworkDefinition network;
    network.SetConvolutionType(ConvolutionType::SAME);
    network.AddNode("A", 1);
    network.AddNode("B", 2);
    network.AddEdge("i", {{"A", 0}, {"B", 0}}, Operation::CONVOLUTION, 0);  
    network.AddEdge("j", {{"B", 1}}, Operation::CONTRACTION); 

    Tensor A({4}); 
    A[0] = 1; A[1] = 2; A[2] = 3; A[3] = -5;

    Tensor B({2,1});
    B[{0,0}] = 3; B[{1,0}] = 4;

    Tensor out = network.Evaluate({std::move(A), std::move(B)});

    std::cout << "out order: " << out.Order() << "\n";
    std::cout << "out tensor_size: " << out.TensorSize() << "\n";
    std::cout << "out tensor_size.size(): " << out.TensorSize().size() << "\n";

    std::cout << out.FlatString() << "\n";



    }
*/


    
/*
{

    TensorNetworkDefinition network;
    network.AddNode("A", 1);
    network.AddNode("B", 1);
    network.AddEdge("i", {{"A", 0}, {"B", 0}}, Operation::CONVOLUTION, 0); 
    Tensor A({2}); 
    A[0] = 1; A[1] = 2;

    Tensor B({2});
    B[0] = 3; B[1] = 4;

    Tensor out = network.Evaluate({std::move(A), std::move(B)});

    std::cout << "out order: " << out.Order() << "\n";
    std::cout << "out tensor_size: " << out.TensorSize() << "\n";
    std::cout << "out tensor_size.size(): " << out.TensorSize().size() << "\n";

    std::cout << out.FlatString() << "\n";



}
*/


/*    
    TensorNetworkDefinition network;
    network.AddNode("A", 2);
    network.AddNode("B", 2);
    network.AddEdge("i", {{"A", 0}, {"B", 0}}, Operation::CONVOLUTION, 0); 
    network.AddEdge("j", {{"A", 1}, {"B", 1}}, Operation::CONTRACTION); 
    Tensor A({2,1}); 
    A[{0,0}] = 1; A[{1,0}] = 2;

    Tensor B({2,1});
    B[{0,0}] = 3; B[{1,0}] = 4;

    Tensor out = network.Evaluate({std::move(A), std::move(B)});

    std::cout << "out order: " << out.Order() << "\n";
    std::cout << "out tensor_size: " << out.TensorSize() << "\n";
    std::cout << "out tensor_size.size(): " << out.TensorSize().size() << "\n";

    std::cout << out.FlatString() << "\n";
*/

/*
    TensorNetworkDefinition network;
    network.AddNode("A", 2);
    network.AddNode("B", 2);
    network.AddEdge("i", {{"A", 0}, {"B", 0}}, Operation::CONVOLUTION, 0); 
    network.AddEdge("j", {{"A", 1}, {"B", 1}}, Operation::CONTRACTION); 
    Tensor A({2,2}); 
    A[{0,0}] = 1;  A[{1,0}] = 2;
    A[{0,1}] = -1; A[{1,1}] = -2;

    Tensor B({2,2});
    B[{0,0}] = 3;  B[{1,0}] = 4;
    B[{0,1}] = -5; B[{1,1}] = 10;

    Tensor out = network.Evaluate({std::move(A), std::move(B)});

    std::cout << "out order: " << out.Order() << "\n";
    std::cout << "out tensor_size: " << out.TensorSize() << "\n";
    std::cout << "out tensor_size.size(): " << out.TensorSize().size() << "\n";

    std::cout << out.FlatString() << "\n";
*/


/*
{
    TensorNetworkDefinition network;
    network.AddNode("A", 1);
    network.AddNode("B", 2);
    network.AddNode("C", 1);
    network.AddEdge("i", {{"A", 0}, {"B", 0}}, Operation::CONVOLUTION, 0); 
    network.AddEdge("j", {{"B", 1}, {"C", 0}}, Operation::CONTRACTION, 1);

    Tensor A({2}); 
    A[0] = 1;  A[1] = 2;
    
    Tensor B({2,3});
    B[{0,0}] = 3;  B[{0,1}] = 4;  B[{0,2}] = 6;
    B[{1,0}] = -5; B[{1,1}] = 10; B[{1,2}] = 4;

    Tensor C({3});
    C[0] = 1; C[1] = 5; C[2] = 3;

    Tensor out = network.Evaluate({std::move(A), std::move(B), std::move(C)});

    std::cout << "out order: " << out.Order() << "\n";
    std::cout << "out tensor_size: " << out.TensorSize() << "\n";
    std::cout << "out tensor_size.size(): " << out.TensorSize().size() << "\n";

    std::cout << out.FlatString() << "\n";
}
*/

/*
{
    TensorNetworkDefinition network;
    network.SetConvolutionType(ConvolutionType::SAME);
    network.AddNode("A", 2);
    network.AddNode("B", 3);
    network.AddNode("C", 1);
    network.AddEdge("i", {{"A", 0}, {"B", 0}}, Operation::CONVOLUTION, 0); 
    network.AddEdge("j", {{"A", 1}, {"B", 1}}, Operation::CONTRACTION); 
    network.AddEdge("k", {{"B", 2}, {"C", 0}}, Operation::CONTRACTION, 1);



    Tensor A({2,2}); 
    A[{0,0}] = 1;  A[{1,0}] = 2;
    A[{0,1}] = -1; A[{1,1}] = -2;

    Tensor B({2,2,3});
    B[{0,0,0}] = 3;  B[{1,0,0}] = 4;
    B[{0,1,0}] = -5; B[{1,1,0}] = 10;
    B[{0,1,0}] = -5; B[{1,1,0}] = 10;

    B[{0,0,1}] = 3;  B[{1,0,1}] = 4;
    B[{0,1,1}] = -5; B[{1,1,1}] = 10;
    B[{0,1,1}] = -5; B[{1,1,1}] = 10;

    B[{0,0,2}] = 3;  B[{1,0,2}] = 4;
    B[{0,1,2}] = -5; B[{1,1,2}] = 10;
    B[{0,1,2}] = -5; B[{1,1,2}] = 10;

    Tensor C({3});
    C[0] = 1; C[1] = 5; C[2] = 3;
    
    std::cout << "About to evaluate\n";
    // \todo a copy is happening when I pass the tensors to the vector, but I want to avoid this.
    //       Maybe there's a way to pass a vector of const references
    //       See reference_wrapper, could also simply pass a vector of pointers
    //       Or accept vectors of std::unique_ptr<Tensor>
    Tensor out = network.Evaluate({std::move(A), std::move(B), std::move(C)});
    
    std::cout << "out order: " << out.Order() << "\n";
    std::cout << "out tensor_size: " << out.TensorSize() << "\n";
    std::cout << "out tensor_size.size(): " << out.TensorSize().size() << "\n";

    std::cout << out.FlatString() << "\n";
}
*/



/*
    // \todo check if I can update Tensor to include scalars, and if this will work
    TensorNetworkDefinition network;
    network.AddNode("A", 1);
    network.AddNode("B", 1);
    network.AddEdge("i", {{"A", 0}, {"B", 0}}, Operation::CONTRACTION);

    Tensor A({2});
    A[0] = 18; A[1] = 3;

    Tensor B({2});
    B[0] = 2; B[1] = 8;

    Tensor out = network.Evaluate({std::move(A), std::move(B)});

    std::cout << "out order: " << out.Order() << "\n";
    std::cout << "out tensor_size: " << out.TensorSize() << "\n";
    std::cout << "out tensor_size.size(): " << out.TensorSize().size() << "\n";

    std::cout << out.FlatString() << "\n";
*/
    
/*
    TensorNetworkDefinition network;
    network.AddNode("A", 2); 
    network.AddNode("B", 1);
    network.AddEdge("j", {{"A", 0}}, Operation::CONTRACTION, 0); 
    network.AddEdge("i", {{"A", 1}, {"B", 0}}, Operation::CONTRACTION, 1);

    Tensor A({2,2});
    A[{0,0}] = 18; A[{1,0}] = 3;
    A[{0,1}] = 12; A[{1,1}] = 4;

    Tensor B({2});
    B[0] = 2; B[1] = 8;

    Tensor out = network.Evaluate({std::move(A), std::move(B)});

    std::cout << "out order: " << out.Order() << "\n";
    std::cout << "out tensor_size: " << out.TensorSize() << "\n";
    std::cout << "out tensor_size.size(): " << out.TensorSize().size() << "\n";

    std::cout << out.FlatString() << "\n";
*/

/*
    TensorNetworkDefinition network;
    network.SetConvolutionType(ConvolutionType::CYCLIC);
    network.AddNode("A", 2);
    // \todo should I change the overloads so that if you have an output edge you don't have to 
    //       specify Operation::CONTRACTION?
    network.AddEdge("i", {{"A", 0}}, Operation::CONTRACTION, 0);
    network.AddEdge("j", {{"A", 1}}, Operation::CONTRACTION);
     
   
    Tensor A({2,2});
    A[{0,0}] = 18; A[{1,0}] = 3;
    A[{0,1}] = 12; A[{1,1}] = 4;

    Tensor out = network.Evaluate({std::move(A)});

    std::cout << "out order: " << out.Order() << "\n";
    std::cout << "out tensor_size: " << out.TensorSize() << "\n";
    std::cout << "out tensor_size.size(): " << out.TensorSize().size() << "\n";

    std::cout << out.FlatString() << "\n";
*/


    return 0;
}

