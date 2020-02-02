#ifndef HELPERS_TYPEDEFS_H
#define HELPERS_TYPEDEFS_H
/** 
 * Authors: Geoffrey Sangston - gsangsto@umd.edu
 * Huang Group
 *
 * A miscellaneous collection of generic helper methods, globals, and typedefs. 
 * Subject to reorganization.
 */


#include <iosfwd>  
#include <vector>

namespace OPS {

std::ostream &operator<< (std::ostream &out, const std::vector<size_t>& vec); 

// \todo do we want to worry about both floats and doubles, or should we define
//       a consistent scalar_type and stick to that everywhere?
//       aka 
//       using scalar_type = float; I'm just proceding for now as if we're using float everywhere
extern const float MAX_FLOAT;
extern const float MIN_FLOAT;
extern const float DEFAULT_PRECISION;

bool Equals(float a, float b, float precision = DEFAULT_PRECISION); 

/**
 * Returns the positive modulus of numer mod denom
 */
float PMod(float numer, float denom);


template<typename T>
int Sign(T x) {
    return (T(0) < x) - (x < T(0));
}



} // OPS

#endif // HELPERS_H
