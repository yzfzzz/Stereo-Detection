#pragma once
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <direct.h>
#include "json.hpp"

namespace stereo {
	class MatchAlgorithmBase
	{
	public:
		virtual bool readCameraConfig(std::string path);
		virtual bool stereoRectification();
		virtual bool rgbToGray(cv::Mat& stereo_raw_rgb_image, bool is_show);
		virtual void correct(bool is_show);
		virtual void match(cv::Ptr<cv::StereoMatcher> matcher, std::string show_window_name, bool is_show);

		void imageInit(std::string config_json_path, cv::Mat& stereo_raw_rgb_image, bool is_show) {
			readCameraConfig(config_json_path);
			rgbToGray(stereo_raw_rgb_image, is_show);
			stereoRectification();
			correct(is_show);
		}
		cv::Mat disp, disp_8bit, distance_;
	protected:
		virtual cv::Mat jsonToMat(nlohmann::json json_matrix);
		bool checkFileSignature(const std::string file_path, const std::vector<uint8_t>& expected_signature);
		bool isImageByHeader(const std::string file_path);
		bool isVideoByHeader(const std::string file_path);
		int single_raw_image_width_;
		int single_raw_image_height_;
		cv::Size single_raw_image_size_;

		cv::Mat stereo_raw_rgb_image_;
		cv::Mat left_stereo_raw_rgb_image_;
		cv::Mat right_stereo_raw_rgb_image_;

		cv::Mat left_stereo_raw_gray_image_;
		cv::Mat right_stereo_raw_gray_image_;

		cv::Rect left_stereo_image_select_rect_;
		cv::Rect right_stereo_image_select_rect_;

		cv::Mat left_rectifyed_image_;
		cv::Mat right_rectifyed_image_;

		// 左相机内参矩阵
		cv::Mat left_camera_matrix_;
		// 左相机畸变参数
		cv::Mat left_camera_distcoeff_;
		// 右相机内参矩阵
		cv::Mat right_camera_matrix_;
		// 左相机畸变参数
		cv::Mat right_camera_distcoeff_;
		// 平移向量
		cv::Mat translation_vector_;
		// 旋转向量
		cv::Mat rotation_vector_;
		// 旋转矩阵
		cv::Mat rotation_matrix_;

		cv::Rect left_valid_roi_;
		cv::Rect right_valid_roi_;

		cv::Mat left_map_x_;
		cv::Mat left_map_y_;
		cv::Mat right_map_x_;
		cv::Mat right_map_y_;

		cv::Mat left_correct_rotation_matrix_;
		cv::Mat right_correct_rotation_matrix_;

		cv::Mat left_project_matrix_;
		cv::Mat right_project_matrix_;

		cv::Mat reproject_matrix_;
	};

	class BM_Algorithm : public MatchAlgorithmBase {
	public:
		BM_Algorithm(int block_size = 21, int min_disparity = 0, int ndisparities = 64, int uniqueness_ratio = 0, int pre_filter_cap = 31,
			int speckle_window_size = 100, int speckle_range = 32, int disp12_max_diff = -1);
		void run(std::string show_window_name, std::string config_json_path = "camera_config.json",
			std::string file_path = "car.jpg", bool is_show = true);
	private:
		cv::Ptr<cv::StereoBM> bm_;
	};

	class SGBM_Algorithm : public MatchAlgorithmBase
	{
	public:
		SGBM_Algorithm(int block_size = 8, int min_disparity = 1, int ndisparities = 64, int image_channels = 3, int uniqueness_ratio = 10,
			int pre_filter_cap = 1, int speckle_window_size = 100, int speckle_range = 100, int disp12_max_diff = -1);
		void run(std::string show_window_name, std::string config_json_path = "camera_config.json",
			std::string file_path = "car.jpg", bool is_show = true);

	private:
		cv::Ptr<cv::StereoSGBM> sgbm_;
	};

}; // namespace stereo

extern int test1();
