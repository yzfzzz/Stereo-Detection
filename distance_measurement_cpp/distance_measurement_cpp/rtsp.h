#pragma once
#include <opencv2/opencv.hpp>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}
class RTSP {
public:
	AVFrame* convertMatToAVFrame(const cv::Mat& mat) {
		// 1. 初始化 SwsContext
		SwsContext* swsCtx = sws_getContext(
			mat.cols, mat.rows, AV_PIX_FMT_BGR24,
			mat.cols, mat.rows, AV_PIX_FMT_YUV420P,
			SWS_BICUBIC, nullptr, nullptr, nullptr);

		// 2. 创建并分配 AVFrame
		AVFrame* frame = av_frame_alloc();
		frame->width = mat.cols;
		frame->height = mat.rows;
		frame->format = AV_PIX_FMT_YUV420P;
		av_frame_get_buffer(frame, 0);

		// 3. 执行转换
		int cvLinesize[1] = { static_cast<int>(mat.step) };
		sws_scale(swsCtx,
			&mat.data, cvLinesize, 0, mat.rows,
			frame->data, frame->linesize);

		// 4. 清理
		sws_freeContext(swsCtx);
		return frame;
	}

};
