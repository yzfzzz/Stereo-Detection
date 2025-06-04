#include "stereo_match_algorithm.h"
#include "mouse_controller.h"
#include <vector>
#include <math.h>

namespace stereo {

	bool MatchAlgorithmBase::checkFileSignature(const std::string file_path, const std::vector<uint8_t>& expected_signature) {
		std::ifstream file(file_path, std::ios::binary);
		if (!file) return false;

		std::vector<uint8_t> buffer(expected_signature.size());
		file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
		return buffer == expected_signature;
	}

	bool MatchAlgorithmBase::isImageByHeader(const std::string file_path) {
		static const std::vector<std::pair<std::vector<uint8_t>, std::string>> image_signatures = {
			{{0xFF, 0xD8, 0xFF}, "JPEG"},  // JPEG文件头
			{{0x89, 0x50, 0x4E, 0x47}, "PNG"},  // PNG文件头
			{{0x42, 0x4D}, "BMP"}  // BMP文件头
		};
		for (const auto& sig : image_signatures) {
			if (checkFileSignature(file_path, sig.first)) return true;
		}
		return false;
	}

	bool MatchAlgorithmBase::isVideoByHeader(const std::string file_path) {
		// 视频文件头通常较复杂，需结合具体格式（如MP4的ftyp盒子）
		std::ifstream file(file_path, std::ios::binary);
		if (!file) return false;

		char buffer[8];
		file.read(buffer, 8);
		// 检查MP4的"ftyp"标识（简化版）
		return std::string(buffer + 4, 4) == "ftyp";
	}

	cv::Mat MatchAlgorithmBase::jsonToMat(nlohmann::json json_matrix) {
		// 提取矩阵数据
		size_t rows = json_matrix.size();
		size_t cols = (rows > 0) ? json_matrix[0].size() : 0;

		// 转换为 cv::Mat
		cv::Mat mat(rows, cols, CV_64F);
		for (size_t i = 0; i < rows; ++i) {
			for (size_t k = 0; k < cols; ++k) {

				mat.at<double>(i, k) = json_matrix[i][k].get<double>();
				//std::cout << std::setprecision(10) << mat.at<double>(i, k) << std::endl;
			}
		}
		return mat;
	}

	bool MatchAlgorithmBase::readCameraConfig(std::string path) {
		std::ifstream file(path);
		if (!file.is_open()) {
			std::cerr << "无法打开 json 文件" << std::endl;
			return -1;
		}

		nlohmann::json config_json;
		try {
			file >> config_json;
		}
		catch (const nlohmann::json::parse_error& e) {
			std::cerr << "JSON 解析错误: " << e.what() << std::endl;
			return -1;
		}
		left_camera_matrix_ = jsonToMat(config_json["left_camera_matrix"]);
		right_camera_matrix_ = jsonToMat(config_json["right_camera_matrix"]);
		left_camera_distcoeff_ = jsonToMat(config_json["left_camera_distcoeff"]);
		right_camera_distcoeff_ = jsonToMat(config_json["right_camera_distcoeff"]);
		translation_vector_ = jsonToMat(config_json["translation_vector"]);
		rotation_vector_ = jsonToMat(config_json["rotation_vector"]);
	}

	bool MatchAlgorithmBase::stereoRectification() {
		cv::Rodrigues(rotation_vector_, rotation_matrix_);

		cv::stereoRectify(left_camera_matrix_, left_camera_distcoeff_, right_camera_matrix_, right_camera_distcoeff_,
			single_raw_image_size_, rotation_matrix_, translation_vector_, left_correct_rotation_matrix_,
			right_correct_rotation_matrix_, left_project_matrix_, right_project_matrix_, reproject_matrix_,
			cv::CALIB_ZERO_DISPARITY, 0, single_raw_image_size_, &left_valid_roi_, &right_valid_roi_);
		cv::initUndistortRectifyMap(left_camera_matrix_, left_camera_distcoeff_, left_correct_rotation_matrix_,
			left_project_matrix_, single_raw_image_size_,
			CV_32FC1, left_map_x_, left_map_y_);
		cv::initUndistortRectifyMap(right_camera_matrix_, right_camera_distcoeff_, right_correct_rotation_matrix_,
			right_project_matrix_, single_raw_image_size_,
			CV_32FC1, right_map_x_, right_map_y_);
		return true;
	}

	bool MatchAlgorithmBase::rgbToGray(cv::Mat& stereo_raw_rgb_image, bool is_show) {
		stereo_raw_rgb_image_ = stereo_raw_rgb_image;
		if (!stereo_raw_rgb_image_.empty()) {
			single_raw_image_height_ = stereo_raw_rgb_image_.rows;
			single_raw_image_width_ = stereo_raw_rgb_image_.cols / 2;
			single_raw_image_size_ = cv::Size(single_raw_image_width_, single_raw_image_height_);
			left_stereo_image_select_rect_ = cv::Rect(0, 0, single_raw_image_width_, single_raw_image_height_);
			right_stereo_image_select_rect_ = cv::Rect(single_raw_image_width_, 0, single_raw_image_width_, single_raw_image_height_);
		}
		else {
			std::cerr << "image file is empty!" << std::endl;
			return false;
		}
		left_stereo_raw_rgb_image_ = stereo_raw_rgb_image_(left_stereo_image_select_rect_);
		right_stereo_raw_rgb_image_ = stereo_raw_rgb_image_(right_stereo_image_select_rect_);

		cv::cvtColor(left_stereo_raw_rgb_image_, left_stereo_raw_gray_image_, cv::COLOR_BGR2GRAY);
		cv::cvtColor(right_stereo_raw_rgb_image_, right_stereo_raw_gray_image_, cv::COLOR_BGR2GRAY);
		return true;
	}

	void MatchAlgorithmBase::correct(bool is_show) {
		cv::remap(left_stereo_raw_gray_image_, left_rectifyed_image_, left_map_x_,
			left_map_y_, cv::INTER_LINEAR);
		cv::remap(right_stereo_raw_gray_image_, right_rectifyed_image_, right_map_x_,
			right_map_y_, cv::INTER_LINEAR);
	}

	void MatchAlgorithmBase::match(cv::Ptr<cv::StereoMatcher> matcher, std::string show_window_name, bool is_show) {
		matcher->compute(left_rectifyed_image_, right_rectifyed_image_, disp);
		disp_8bit = cv::Mat(disp.rows, disp.cols, CV_8UC1);
		cv::normalize(disp, disp_8bit, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		cv::reprojectImageTo3D(disp, distance_, reproject_matrix_, is_show);
		distance_ = distance_ * 16;

		if (is_show) {
			cv::imshow(show_window_name, disp_8bit);
		}
	}

	SGBM_Algorithm::SGBM_Algorithm(int block_size, int min_disparity, int ndisparities, int image_channels, int	uniqueness_ratio,
		int pre_filter_cap, int speckle_window_size, int speckle_range, int disp12_max_diff) {
		sgbm_ = cv::StereoSGBM::create(min_disparity, ndisparities, block_size);
		int p1 = 8 * image_channels * block_size * block_size;
		int p2 = 32 * image_channels * block_size * block_size;
		sgbm_->setP1(p1);
		sgbm_->setP2(p2);
		sgbm_->setPreFilterCap(pre_filter_cap);
		sgbm_->setUniquenessRatio(uniqueness_ratio);
		sgbm_->setSpeckleRange(speckle_range);
		sgbm_->setSpeckleWindowSize(speckle_window_size);
		sgbm_->setDisp12MaxDiff(disp12_max_diff);
		sgbm_->setMode(cv::StereoSGBM::MODE_HH);
	};

	void SGBM_Algorithm::run(std::string show_window_name, std::string config_json_path, std::string file_path, bool is_show) {
		if (isImageByHeader(file_path)) {
			cv::Mat frame = cv::imread(file_path);
			imageInit(config_json_path, frame, is_show);
			match(sgbm_, show_window_name, is_show);
			cv::waitKey();
		}
		else {
			std::cout << "video" << std::endl;
			// 打开视频文件（或摄像头：VideoCapture cap(0);）
			cv::VideoCapture cap(file_path);
			if (!cap.isOpened()) {
				std::cerr << "Error: Could not open video file!" << std::endl;
				return;
			}

			// 获取视频属性（可选）
			double fps = cap.get(cv::CAP_PROP_FPS);
			int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
			int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
			std::cout << "FPS: " << fps << ", Resolution: " << width << "x" << height << std::endl;

			// 逐帧读取
			cv::Mat frame;
			while (cap.read(frame)) {  // 等效于 cap >> frame;
				if (frame.empty()) break;
				imageInit(config_json_path, frame, is_show);
				match(sgbm_, show_window_name, is_show);

				if (cv::waitKey(30) == 27) break;  // 按ESC键退出
			}

			cap.release();
			cv::destroyAllWindows();

		}
	}

	BM_Algorithm::BM_Algorithm(int block_size, int min_disparity, int ndisparities, int uniqueness_ratio, int pre_filter_cap,
		int speckle_window_size, int speckle_range, int disp12_max_diff) {
		if (block_size % 2 == 0) {
			block_size++;
		}
		bm_ = cv::StereoBM::create(ndisparities, block_size);
		bm_->setROI1(left_valid_roi_);
		bm_->setROI2(right_valid_roi_);
		bm_->setPreFilterCap(pre_filter_cap);
		bm_->setMinDisparity(min_disparity);  //最小视差，默认值为0, 可以是负值，int型
		bm_->setUniquenessRatio(uniqueness_ratio);//uniquenessRatio主要可以防止误匹配
		bm_->setSpeckleWindowSize(speckle_window_size);
		bm_->setSpeckleRange(speckle_range);
		bm_->setDisp12MaxDiff(disp12_max_diff);
	}

	void BM_Algorithm::run(std::string show_window_name, std::string config_json_path, std::string file_path, bool is_show) {
		if (isImageByHeader(file_path)) {
			cv::Mat frame = cv::imread(file_path);
			imageInit(config_json_path, frame, is_show);
			match(bm_, show_window_name, is_show);
			cv::waitKey();
		}
		else {
			std::cout << "video" << std::endl;
			// 打开视频文件（或摄像头：VideoCapture cap(0);）
			cv::VideoCapture cap(file_path);
			if (!cap.isOpened()) {
				std::cerr << "Error: Could not open video file!" << std::endl;
				return;
			}

			// 获取视频属性（可选）
			double fps = cap.get(cv::CAP_PROP_FPS);
			int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
			int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
			std::cout << "FPS: " << fps << ", Resolution: " << width << "x" << height << std::endl;

			// 逐帧读取
			cv::Mat frame;
			while (cap.read(frame)) {  // 等效于 cap >> frame;
				if (frame.empty()) break;
				imageInit(config_json_path, frame, is_show);
				match(bm_, show_window_name, is_show);

				if (cv::waitKey(30) == 27) break;  // 按ESC键退出
			}

			cap.release();
			cv::destroyAllWindows();
		}
	}

};  // namespace stereo


int test1() {

	char buffer[_MAX_PATH];
	if (_getcwd(buffer, _MAX_PATH) != nullptr) {
		std::cout << "当前工作目录: " << buffer << std::endl;
	}
	else {
		std::cerr << "获取路径失败！" << std::endl;
	}

	stereo::SGBM_Algorithm stereo_algorithm;
	std::string show_window_name = "disp8";
	cv::namedWindow(show_window_name, cv::WINDOW_AUTOSIZE);
	MouseController mouseController(show_window_name);
	mouseController.registerCallback(MouseController::MouseEventType::LBUTTON_DOWN,
		[&](int x, int y, int flags, void* userdata) {
			if (static_cast<MouseController*>(userdata)->isButtonDown(MouseController::MouseEventType::LBUTTON_DOWN)) {
				cv::Vec3f point3d = stereo_algorithm.distance_.at<cv::Vec3f>(cv::Point(x, y));
				double d = 0;
				for (int i = 0; i < 3; i++) {
					d += point3d[i] * point3d[i];
				}
				d = sqrt(d);
				d = d / 10.0;
				printf("(%d, %d) dis: %.2lf cm \r\n", x, y, d);
			}
		});

	stereo_algorithm.run(show_window_name, "camera_config.json", "./car.avi");
	return 0;
}