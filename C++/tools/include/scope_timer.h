
#include <chrono>
#include <iostream>
#include <string>
#include <map>


class ScopedTimer {
  public:
    explicit ScopedTimer(std::string name) : name_(std::move(name)), start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::steady_clock::now();
        auto us  = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        // 获取局部静态变量的引用，确保初始化顺序可控
        auto& timers = GetScopedTimers(); 
        if (timers.find(name_) != timers.end()) {
            timers[name_] += us;
        } else {
            timers[name_] = us;
        }
    }

        // 替换原来的类外静态成员，使用 Meyers' Singleton
    static std::map<std::string, double>& GetScopedTimers() {
        static std::map<std::string, double> scoped_timers = {
            { "1.Cap read",               0.0 },
            { "2.YOLO inference",         0.0 },
            { "3.Depth inference",        0.0 },
            { "4.ByteTrack",              0.0 },
            { "5.Draw",                   0.0 },
            { "6.Write",                  0.0 },
            { "One frame average time", 0.0 },
        };
        return scoped_timers;
    }

  private:
    std::string                           name_;
    std::chrono::steady_clock::time_point start_;

};
// 删除原来的类外定义: std::unordered_map<...> ScopedTimer::scoped_timers = ...;

