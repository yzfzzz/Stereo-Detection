#include "mouse_controller.h"

MouseController::MouseController(const std::string& windowName)
	: m_windowName(windowName) {
	cv::setMouseCallback(m_windowName, &MouseController::onMouse, this);
}

void MouseController::registerCallback(MouseEventType eventType, MouseCallback callback) {
	m_callbacks[eventType] = callback;
}

void MouseController::removeCallback(MouseEventType eventType) {
	m_callbacks.erase(eventType);
}

cv::Point MouseController::getLastPosition() const {
	return m_lastPosition;
}

bool MouseController::isButtonDown(MouseEventType button) const {
	switch (button) {
	case MouseEventType::LBUTTON_DOWN: return m_lButtonDown;
	case MouseEventType::RBUTTON_DOWN: return m_rButtonDown;
	case MouseEventType::MBUTTON_DOWN: return m_mButtonDown;
	default: return false;
	}
}

void MouseController::onMouse(int event, int x, int y, int flags, void* userdata) {
	MouseController* controller = static_cast<MouseController*>(userdata);
	controller->handleMouseEvent(event, x, y, flags);
}

void MouseController::handleMouseEvent(int event, int x, int y, int flags) {
	m_lastPosition = cv::Point(x, y);

	// 更新按钮状态
	switch (event) {
	case cv::EVENT_LBUTTONDOWN: m_lButtonDown = true; break;
	case cv::EVENT_RBUTTONDOWN: m_rButtonDown = true; break;
	case cv::EVENT_MBUTTONDOWN: m_mButtonDown = true; break;
	case cv::EVENT_LBUTTONUP: m_lButtonDown = false; break;
	case cv::EVENT_RBUTTONUP: m_rButtonDown = false; break;
	case cv::EVENT_MBUTTONUP: m_mButtonDown = false; break;
	}

	// 调用注册的回调函数
	auto it = m_callbacks.find(static_cast<MouseEventType>(event));
	if (it != m_callbacks.end()) {
		it->second(x, y, flags, this);
	}
}
