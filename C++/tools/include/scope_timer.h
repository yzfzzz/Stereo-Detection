
#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

std::unordered_map<std::string, double> scoped_timers = {
    { "Cap read",               0.0 },
    { "YOLO inference",         0.0 },
    { "Depth inference",        0.0 },
    { "ByteTrack",              0.0 },
    { "Draw",                   0.0 },
    { "Write",                  0.0 },
    { "One frame average time", 0.0 },
};

class ScopedTimer {
  public:
    explicit ScopedTimer(std::string name) : name_(std::move(name)), start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::steady_clock::now();
        auto us  = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        if (scoped_timers.find(name_) != scoped_timers.end()) {
            scoped_timers[name_] += us;
        } else {
            scoped_timers[name_] = us;
        }
    }

  private:
    std::string                           name_;
    std::chrono::steady_clock::time_point start_;
};
