#include <iostream>

#include <cmath>
#include <ostream>
#include <limits>
#include <numeric>

#include "HelpersTypedefs.h"


namespace OPS {

const float MAX_FLOAT = std::numeric_limits<float>::max();
const float MIN_FLOAT = std::numeric_limits<float>::lowest();
const float DEFAULT_PRECISION = 1e-5;

std::ostream &operator<< (std::ostream &out, const std::vector<size_t>& vec) {
    if(vec.size() == 0) {
        return out;
    }
	for(size_t i = 0; i < vec.size(); i++) {
	    out << vec[i]; 
        if(i < vec.size()-1) {
            out << " ";
        }
    }
	return out; 
}


bool Equals(float a, float b, float precision) {
    return std::abs(a - b) < precision;
}

float PMod(float numer, float denom) {
    return std::fmod(std::fmod(numer, denom) + denom, denom);
}


size_t MultiplyElements(const std::vector<size_t>& iterable) {
    if(iterable.size() == 0) {
        std::cout << "returning 1\n";
        return 1;
    }
    return (std::accumulate(std::begin(iterable), std::end(iterable), 1, std::multiplies<float>()));
}

} // OPS
