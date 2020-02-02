#include <iostream>
#include <cmath>

#include "Profiling.h"



namespace OPS {

void Prof::Mark(int id, std::string to_print, bool print_each_call) {
    
    using namespace std::chrono;
    auto now = steady_clock::now();

	if(last_timestamp.count(id) == 0) { 
        last_timestamp[id] = now; 
	} else {
        auto last = last_timestamp[id];
        last_timestamp.erase(id);
        // we only want to show the elapsed time between the first and second... I think
        // might want to show the outside elapsed ticks but I don't think so

        size_t elapsed = duration_cast<microseconds>(now - last).count(); 

        if(print_each_call) {
            std::cout << id << ", elapsed microseconds = " 
                            << elapsed << " - " << to_print << "\n";
        }

        if(info.count(id) == 0) {
            info[id] = {1, elapsed, elapsed, double(elapsed), to_print};
        } else {

            ProfileInfo old_info = info[id];
            old_info.num_profiles++;
            old_info.last_micros = elapsed;
            old_info.max_micros = std::max(old_info.max_micros, elapsed);
            old_info.avg_micros 
             = ((old_info.num_profiles-1)*old_info.avg_micros + elapsed)/old_info.num_profiles;
            old_info.msg = to_print;
            info[id] = old_info;
            
        }
    }
    
}


void Prof::PrintInfo() {
    for(auto it = info.begin(); it != info.end(); ++it) {
        PrintInfo(it->first);
    }

}

void Prof::PrintInfo(int id) {
    std::cout << id << ", " << info[id].msg
                    << ", num profiles = " << info[id].num_profiles
                    << ", last elapsed microseconds = " << info[id].last_micros
                    << ", max elapsed microseconds = " << info[id].max_micros
                    << ", avg elapsed microseconds = " << info[id].avg_micros << "\n";

}



} // OPS

