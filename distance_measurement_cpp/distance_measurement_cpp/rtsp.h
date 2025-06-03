#pragma once
#include <opencv2/opencv.hpp>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}
class RTSP {
public:
	AVFrame* convertMatToAVFrame(const cv::Mat& mat) {
		// 1. ��ʼ�� SwsContext
		SwsContext* swsCtx = sws_getContext(
			mat.cols, mat.rows, AV_PIX_FMT_BGR24,
			mat.cols, mat.rows, AV_PIX_FMT_YUV420P,
			SWS_BICUBIC, nullptr, nullptr, nullptr);

		// 2. ���������� AVFrame
		AVFrame* frame = av_frame_alloc();
		frame->width = mat.cols;
		frame->height = mat.rows;
		frame->format = AV_PIX_FMT_YUV420P;
		av_frame_get_buffer(frame, 0);

		// 3. ִ��ת��
		int cvLinesize[1] = { static_cast<int>(mat.step) };
		sws_scale(swsCtx,
			&mat.data, cvLinesize, 0, mat.rows,
			frame->data, frame->linesize);

		// 4. ����
		sws_freeContext(swsCtx);
		return frame;
	}

};
