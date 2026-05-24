#include "scope_timer.h"

ScopedTimer::~ScopedTimer() {
    auto   end = std::chrono::steady_clock::now();
    auto   us  = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
    auto & timers_table = GetScopedTimers();
    timers_table[name_].push_back(us);
}

std::map<std::string, std::vector<double>> & ScopedTimer::GetScopedTimers() {
    static std::map<std::string, std::vector<double>> scoped_timers_table;
    return scoped_timers_table;
}
