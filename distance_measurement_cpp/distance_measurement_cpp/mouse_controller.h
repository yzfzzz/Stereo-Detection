#pragma once
#include <opencv2/opencv.hpp>
#include <functional>
#include <map>

class MouseController {
public:
	// ����¼�����ö��
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

	// ���ص���������
	using MouseCallback = std::function<void(int x, int y, int flags, void* userdata)>;

	// ���캯��
	explicit MouseController(const std::string& windowName);

	// ע������¼��ص�
	void registerCallback(MouseEventType eventType, MouseCallback callback);

	// �Ƴ�����¼��ص�
	void removeCallback(MouseEventType eventType);

	// ��ȡ������λ��
	cv::Point getLastPosition() const;

	// ��ȡ��갴��״̬
	bool isButtonDown(MouseEventType button) const;

private:
	// ��̬���ص�����
	static void onMouse(int event, int x, int y, int flags, void* userdata);

	// ʵ����괦����
	void handleMouseEvent(int event, int x, int y, int flags);

	std::string m_windowName;
	std::map<MouseEventType, MouseCallback> m_callbacks;
	cv::Point m_lastPosition;
	bool m_lButtonDown = false;
	bool m_rButtonDown = false;
	bool m_mButtonDown = false;
};