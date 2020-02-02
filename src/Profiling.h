#ifndef PROFILING_H
#define PROFILING_H

/** 
 * Authors: Geoffrey Sangston - gsangsto@umd.edu
 * Huang Group
 *
 * A provisional profiler for making rough estimates 
 */

#include <map>
#include <string>
#include <chrono>

namespace OPS {

class Prof {

    public:

    /**
     * To be used like:
     *
     * ops_prof.Mark(0);
     * ... lots of code
     * ops_prof.Mark(0, "From ... lots of code");
     * ops_prof.PrintInfo(0);
     * ops_prof.Mark(1);
     * ... lots of code 2 
     * ops_prof.Mark(1, "From ... lots of code 2");
     * ops_prof.PrintInfo(1)
     *
     */
    void Mark(int id = 0, std::string to_print = "", bool print_each_call = false);
    void PrintInfo();
    void PrintInfo(int id);


    private: 

    std::map<int, std::chrono::time_point<std::chrono::steady_clock>> last_timestamp;

    struct ProfileInfo {
        size_t num_profiles = 0;

        size_t last_micros = 0;
        size_t max_micros = 0; 
        double avg_micros = 0;

        std::string msg;

    };

    std::map<int, ProfileInfo> info;
    
}; // Prof

} // OPS

extern OPS::Prof ops_prof;

#endif // PROFILING_H

