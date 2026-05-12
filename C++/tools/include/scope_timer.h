#pragma once
#include <chrono>
#include <map>
#include <string>
#include <utility>

class ScopedTimer {
  public:
    explicit ScopedTimer(std::string name) : name_(std::move(name)), start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer();

    static std::map<std::string, double> & GetScopedTimers();

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
