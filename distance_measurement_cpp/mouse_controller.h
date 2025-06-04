#pragma once
#include <opencv2/opencv.hpp>
#include <functional>
#include <map>

class MouseController {
public:
	// 鼠标事件类型枚举
	enum class MouseEventType {
		LBUTTON_DOWN = cv::EVENT_LBUTTONDOWN,
		RBUTTON_DOWN = cv::EVENT_RBUTTONDOWN,
		MBUTTON_DOWN = cv::EVENT_MBUTTONDOWN,
		LBUTTON_UP = cv::EVENT_LBUTTONUP,
		RBUTTON_UP = cv::EVENT_RBUTTONUP,
		MBUTTON_UP = cv::EVENT_MBUTTONUP,
		MOUSE_MOVE = cv::EVENT_MOUSEMOVE,
		LBUTTON_DBLCLK = cv::EVENT_LBUTTONDBLCLK,
		RBUTTON_DBLCLK = cv::EVENT_RBUTTONDBLCLK,
		MBUTTON_DBLCLK = cv::EVENT_MBUTTONDBLCLK
	};

	// 鼠标回调函数类型
	using MouseCallback = std::function<void(int x, int y, int flags, void* userdata)>;

	// 构造函数
	explicit MouseController(const std::string& windowName);

	// 注册鼠标事件回调
	void registerCallback(MouseEventType eventType, MouseCallback callback);

	// 移除鼠标事件回调
	void removeCallback(MouseEventType eventType);

	// 获取最后鼠标位置
	cv::Point getLastPosition() const;

	// 获取鼠标按下状态
	bool isButtonDown(MouseEventType button) const;

private:
	// 静态鼠标回调函数
	static void onMouse(int event, int x, int y, int flags, void* userdata);

	// 实例鼠标处理函数
	void handleMouseEvent(int event, int x, int y, int flags);

	std::string m_windowName;
	std::map<MouseEventType, MouseCallback> m_callbacks;
	cv::Point m_lastPosition;
	bool m_lButtonDown = false;
	bool m_rButtonDown = false;
	bool m_mButtonDown = false;
};