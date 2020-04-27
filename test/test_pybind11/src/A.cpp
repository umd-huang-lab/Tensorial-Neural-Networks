#include <iostream>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

int add(int i, int j) {
    return i + j;
}


void PrintString(std::string str) {
    std::cout << str << "\n";
}

size_t AddAll(std::vector<size_t> nums) {
    size_t res = 0;
    
    for(size_t i = 0; i < nums.size(); i++) {
        res += nums[i];   
    }

    return res;
}


// the first argument to PYBIND11_MODULE must agree with the name of the shared object library
PYBIND11_MODULE(test_python_binding, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");

    m.def("PrintString", &PrintString, "Print String");

    m.def("AddAll", &AddAll, "Add All");
}

