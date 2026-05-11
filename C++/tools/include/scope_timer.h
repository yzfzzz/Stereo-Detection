#pragma once
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <utility>

class ScopedTimer {
  public:
    explicit ScopedTimer(std::string name) : name_(std::move(name)), start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        auto   end          = std::chrono::steady_clock::now();
        auto   us           = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        auto & timers_table = GetScopedTimers();
        if (timers_table.find(name_) != timers_table.end()) {
            timers_table[name_] += us;
        } else {
            timers_table[name_] = us;
        }
    }

    static std::map<std::string, double> & GetScopedTimers() {
        static std::map<std::string, double> scoped_timers_table;
        return scoped_timers_table;
    }

  private:
    std::string                           name_;
    std::chrono::steady_clock::time_point start_;
};

// 基础实现：接受任意可调用对象
template <typename Func>
auto DEBUG_FUNCTION_RUNNING_TIME_IMPL(const std::string & name, Func && func) -> decltype(func()) {
    ScopedTimer timer(name);
    return func();
}

// 版本1：自定义名称 + 可调用对象（适合 lambda）
#define DEBUG_FUNCTION_RUNNING_TIME(name, func) DEBUG_FUNCTION_RUNNING_TIME_IMPL(name, func)

// 版本2：自动函数名 + 可调用对象（适合 lambda）
#define DEBUG_FUNCTION_RUNNING_TIME_AUTO(func) DEBUG_FUNCTION_RUNNING_TIME_IMPL(#func, func)

// 版本3：成员函数专用（对象指针）
#define DEBUG_FUNCTION_RUNNING_TIME_MEMBER_PTR(name, obj, method, ...) \
    DEBUG_FUNCTION_RUNNING_TIME_IMPL(name, [&]() { return (obj)->method(__VA_ARGS__); })

// 版本4：成员函数专用（对象引用）
#define DEBUG_FUNCTION_RUNNING_TIME_MEMBER_REF(name, obj, method, ...) \
    DEBUG_FUNCTION_RUNNING_TIME_IMPL(name, [&]() { return (obj).method(__VA_ARGS__); })

// 版本5：自由函数带参数版本
#define DEBUG_FUNCTION_RUNNING_TIME_FUNC(name, func, ...) \
    DEBUG_FUNCTION_RUNNING_TIME_IMPL(name, [&]() { return func(__VA_ARGS__); })
