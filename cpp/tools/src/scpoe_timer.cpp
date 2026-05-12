#include "scope_timer.h"

ScopedTimer::~ScopedTimer() {
    auto   end          = std::chrono::steady_clock::now();
    auto   us           = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
    auto & timers_table = GetScopedTimers();
    if (timers_table.find(name_) != timers_table.end()) {
        timers_table[name_] += us;
    } else {
        timers_table[name_] = us;
    }
}

std::map<std::string, double> & ScopedTimer::GetScopedTimers() {
    static std::map<std::string, double> scoped_timers_table;
    return scoped_timers_table;
}
