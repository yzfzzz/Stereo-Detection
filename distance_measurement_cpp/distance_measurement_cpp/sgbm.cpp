#include <math.h>

#include <iostream>
#include <opencv2/opencv.hpp>

#include "mouse_controller.h"
#include "stereo_match_algorithm.h"
//
//const int imageWidth = 640;  // ����ͷ�ķֱ���
//const int imageHeight = 480;
//cv::Vec3f point3;
//float d;
//cv::Size imageSize = cv::Size(imageWidth, imageHeight);
//
//cv::Mat img;
//cv::Mat rgbImageL, grayImageL;
//cv::Mat rgbImageR, grayImageR;
//cv::Mat rectifyImageL, rectifyImageR;
//cv::Rect m_l_select;
//cv::Rect m_r_select;
//
//cv::Rect
//validROIL;  // ͼ��У��֮�󣬻��ͼ����вü��������validROI����ָ�ü�֮�������
//cv::Rect validROIR;
//
//cv::Mat mapLx, mapLy, mapRx, mapRy;  // ӳ���
//cv::Mat Rl, Rr, Pl, Pr, Q;           // У����ת����R��ͶӰ����P ��ͶӰ����Q
//cv::Mat xyz;                         // ��ά����
//
//cv::Point origin;           // ��갴�µ���ʼ��
//cv::Rect selection;         // �������ѡ��
//bool selectObject = false;  // �Ƿ�ѡ�����
//
//int blockSize = 8, mindisparity = 1, ndisparities = 64, img_channels = 3;
//cv::Ptr<cv::StereoSGBM> sgbm =
//cv::StereoSGBM::create(mindisparity, ndisparities, blockSize);
//
///*���ȱ궨�õ���������ڲξ���
//fx 0 cx
//0 fy cy
//0  0  1
//*/
//cv::Mat cameraMatrixL = (cv::Mat_<double>(3, 3) << 516.5066236, -1.444673028,
//	320.2950423, 0, 516.5816117, 270.7881873, 0, 0, 1.);
//// ��õĻ������
//
///*418.523322187048	0	0
//-1.26842201390676	421.222568242056	0
//344.758267538961	243.318992284899	1 */ //2
//
//cv::Mat distCoeffL = (cv::Mat_<double>(5, 1) << -0.046645194, 0.077595167,
//	0.012476819, -0.000711358, 0);
////[0.006636837611004,0.050240447649195] [0.006681263320267,0.003130367429418]
//
///*���ȱ궨�õ���������ڲξ���
//fx 0 cx
//0 fy cy
//0  0  1
//*/
//cv::Mat cameraMatrixR = (cv::Mat_<double>(3, 3) << 511.8428182, 1.295112628,
//	317.310253, 0, 513.0748795, 269.5885026, 0, 0, 1);
//
///*
//417.417985082506	0	0
//0.498638151824367	419.795432389420	0
//309.903372309072	236.256106972796	1
//*/ //2
//
//cv::Mat distCoeffR = (cv::Mat_<double>(5, 1) << -0.061588946, 0.122384376,
//	0.011081232, -0.000750439, 0);
////[-0.038407383078874,0.236392800301615]  [0.004121779274885,0.002296129959664]
//
//cv::Mat T = (cv::Mat_<double>(3, 1) << -120.3559901, -0.188953775,
//	-0.662073075);  // Tƽ������
////[-1.210187345641146e+02,0.519235426836325,-0.425535566316217]
//// ��ӦMatlab����T����
//// Mat rec = (Mat_<double>(3, 1) << -0.00306, -0.03207,
//// 0.00206);//rec��ת��������Ӧmatlab om����  ��
//cv::Mat rec =
//(cv::Mat_<double>(3, 3) << 0.999911333, -0.004351508, 0.012585312,
//	0.004184066, 0.999902792, 0.013300386, -0.012641965, -0.013246549,
//	0.999832341);  // rec��ת��������Ӧmatlab om����  ��
//
///* 0.999341122700880	0.000660748031451783	-0.0362888948713456
//-0.00206388651740061	0.999250989651683	-0.0386419468010579
//0.0362361815232777	0.0386913826603732	0.998593969567432 */
//
//// Mat T = (Mat_<double>(3, 1) << -48.4, 0.241, -0.0344);//Tƽ������
////[-1.210187345641146e+02,0.519235426836325,-0.425535566316217]
//// ��ӦMatlab����T����
//
//cv::Mat R;  // R ��ת����
//
///*****����ƥ��*****/
//void stereo_match(int, void*) {
//	/*
//	bm->setBlockSize(2 * blockSize + 5);     //SAD���ڴ�С��5~21֮��Ϊ��
//	bm->setROI1(validROIL);
//	bm->setROI2(validROIR);
//	bm->setPreFilterCap(31);
//	bm->setMinDisparity(0);  //��С�ӲĬ��ֵΪ0, �����Ǹ�ֵ��int��
//	bm->setNumDisparities(numDisparities * 16 +
//	16);//�Ӳ�ڣ�������Ӳ�ֵ����С�Ӳ�ֵ֮��,���ڴ�С������16����������int��
//	bm->setTextureThreshold(10);
//	bm->setUniquenessRatio(uniquenessRatio);//uniquenessRatio��Ҫ���Է�ֹ��ƥ��
//	bm->setSpeckleWindowSize(100);
//	bm->setSpeckleRange(32);
//	bm->setDisp12MaxDiff(-1);
//	*/
//
//	//int P1 = 8 * img_channels * blockSize * blockSize;
//	//int P2 = 32 * img_channels * blockSize * blockSize;
//	//sgbm->setP1(P1);
//	//sgbm->setP2(P2);
//	//sgbm->setPreFilterCap(1);
//	//sgbm->setUniquenessRatio(10);
//	//sgbm->setSpeckleRange(100);
//	//sgbm->setSpeckleWindowSize(100);
//	//sgbm->setDisp12MaxDiff(-1);
//	//// sgbm->setNumDisparities(1);
//	//sgbm->setMode(cv::StereoSGBM::MODE_HH);
//
//	cv::Mat disp, disp8;
//	sgbm->compute(rectifyImageL, rectifyImageR, disp);  // ����ͼ�����Ϊ�Ҷ�ͼ
//	disp8 = cv::Mat(disp.rows, disp.cols, CV_8UC1);
//	cv::normalize(disp, disp8, 0, 255, cv::NORM_MINMAX, CV_8UC1);
//	cv::reprojectImageTo3D(
//		disp, xyz, Q,
//		true);  // ��ʵ�������ʱ��ReprojectTo3D������X / W, Y / W, Z /
//	// W��Ҫ����16(Ҳ����W����16)�����ܵõ���ȷ����ά������Ϣ��
//	xyz = xyz * 16;
//	cv::imshow("disparity", disp8);
//}
//
///*****�������������ص�*****/
//static void onMouse(int event, int x, int y, int, void*) {
//	if (selectObject) {
//		selection.x = MIN(x, origin.x);
//		selection.y = MIN(y, origin.y);
//		selection.width = std::abs(x - origin.x);
//		selection.height = std::abs(y - origin.y);
//	}
//
//	switch (event) {
//	case cv::EVENT_LBUTTONDOWN:  // �����ť���µ��¼�
//		origin = cv::Point(x, y);
//		selection = cv::Rect(x, y, 0, 0);
//		selectObject = true;
//		// cout << origin << "in world coordinate is: " << xyz.at<Vec3f>(origin)
//		// << endl;
//		point3 = xyz.at<cv::Vec3f>(origin);
//		point3[0];
//		// cout << "point3[0]:" << point3[0] << "point3[1]:" << point3[1] <<
//		// "point3[2]:" << point3[2]<<endl;
//		std::cout << "�������꣺" << std::endl;
//		std::cout << "x: " << point3[0] << "  y: " << point3[1]
//			<< "  z: " << point3[2] << std::endl;
//		d = point3[0] * point3[0] + point3[1] * point3[1] + point3[2] * point3[2];
//		d = sqrt(d);  // mm
//		// cout << "������:" << d << "mm" << endl;
//
//		d = d / 10.0;  // cm
//		std::cout << "������:" << d << "cm" << std::endl;
//
//		// d = d/1000.0;   //m
//		// cout << "������:" << d << "m" << endl;
//
//		break;
//	case cv::EVENT_LBUTTONUP:  // �����ť�ͷŵ��¼�
//		selectObject = false;
//		if (selection.width > 0 && selection.height > 0) break;
//	}
//}
//
///*****������*****/
//int test2() {
//	/*
//	����У��
//	*/
//	cv::Rodrigues(rec, R);  // Rodrigues�任
//	cv::stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR,
//		imageSize, R, T, Rl, Rr, Pl, Pr, Q,
//		cv::CALIB_ZERO_DISPARITY, 0, imageSize, &validROIL,
//		&validROIR);
//	cv::initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize,
//		CV_32FC1, mapLx, mapLy);
//	cv::initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize,
//		CV_32FC1, mapRx, mapRy);
//
//	/*
//	��ȡͼƬ
//	*/
//
//	m_l_select = cv::Rect(0, 0, 640, 480);
//	img = cv::imread("car.jpg", cv::IMREAD_COLOR);
//	// imshow("Image", img);
//	rgbImageL = img(m_l_select);
//	cvtColor(rgbImageL, grayImageL, cv::COLOR_BGR2GRAY);
//
//	m_r_select = cv::Rect(640, 0, 640, 480);
//	rgbImageR = img(m_r_select);
//	cvtColor(rgbImageR, grayImageR, cv::COLOR_BGR2GRAY);
//
//	// imshow("ImageL", rgbImageL);
//	// imshow("ImageR", rgbImageR);
//
//	/*
//	����remap֮�����������ͼ���Ѿ����沢���ж�׼��
//	*/
//	cv::remap(grayImageL, rectifyImageL, mapLx, mapLy, cv::INTER_LINEAR);
//	cv::remap(grayImageR, rectifyImageR, mapRx, mapRy, cv::INTER_LINEAR);
//
//	/*
//	��У�������ʾ����
//	*/
//	cv::Mat rgbRectifyImageL, rgbRectifyImageR;
//	cv::cvtColor(rectifyImageL, rgbRectifyImageL,
//		cv::COLOR_GRAY2BGR);  // α��ɫͼ
//	cv::cvtColor(rectifyImageR, rgbRectifyImageR, cv::COLOR_GRAY2BGR);
//
//	// ������ʾ
//	// rectangle(rgbRectifyImageL, validROIL, Scalar(0, 0, 255), 3, 8);
//	// rectangle(rgbRectifyImageR, validROIR, Scalar(0, 0, 255), 3, 8);
//	// imshow("ImageL After Rectify", rgbRectifyImageL);
//	// imshow("ImageR After Rectify", rgbRectifyImageR);
//
//	// ��ʾ��ͬһ��ͼ��
//	cv::Mat canvas;
//	double sf;
//	int w, h;
//	sf = 600. / MAX(imageSize.width, imageSize.height);
//	w = cvRound(imageSize.width * sf);
//	h = cvRound(imageSize.height * sf);
//	canvas.create(h, w * 2, CV_8UC3);  // ע��ͨ��
//
//	// ��ͼ�񻭵�������
//	cv::Mat canvasPart = canvas(cv::Rect(w * 0, 0, w, h));  // �õ�������һ����
//	resize(rgbRectifyImageL, canvasPart, canvasPart.size(), 0, 0,
//		cv::INTER_AREA);  // ��ͼ�����ŵ���canvasPartһ����С
//	cv::Rect vroiL(cvRound(validROIL.x * sf),
//		cvRound(validROIL.y * sf),  // ��ñ���ȡ������
//		cvRound(validROIL.width * sf), cvRound(validROIL.height *
//			sf));
//	// rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8); //����һ������
//	std::cout << "Painted ImageL" << std::endl;
//
//	// ��ͼ�񻭵�������
//	canvasPart = canvas(cv::Rect(w, 0, w, h));  // ��û�������һ����
//	resize(rgbRectifyImageR, canvasPart, canvasPart.size(), 0, 0,
//		cv::INTER_LINEAR);
//	cv::Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y * sf),
//		cvRound(validROIR.width * sf), cvRound(validROIR.height *
//			sf));
//	// rectangle(canvasPart, vroiR, Scalar(0, 0, 255), 3, 8);
//	std::cout << "Painted ImageR" << std::endl;
//
//	// ���϶�Ӧ������
//	for (int i = 0; i < canvas.rows; i += 16)
//		line(canvas, cv::Point(0, i), cv::Point(canvas.cols, i),
//			cv::Scalar(0, 255, 0), 1, 8);
//	imshow("rectified", canvas);
//
//	/*
//	����ƥ��
//	*/
//	cv::namedWindow("disparity", cv::WINDOW_AUTOSIZE);
//
//	/*************************���ο��ӻ�**********************************************/
//	// ����SAD���� Trackbar
//	// createTrackbar("BlockSize:\n", "disparity", &blockSize, 8,
//	// �����Ӳ�Ψһ�԰ٷֱȴ��� Trackbar
//	// createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50,
//	// stereo_match);
//	// �����Ӳ�� Trackbar
//	// createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16,
//	// stereo_match);
//
//	// �����Ӧ����setMouseCallback(��������, ���ص�����,
//	// �����ص������Ĳ�����һ��ȡ0)
//	cv::setMouseCallback("disparity", onMouse, 0);
//	stereo_match(0, 0);
//
//	cv::waitKey();
//	return 0;
//}

/*        ˫Ŀ���        */


using namespace std;
using namespace cv;

const int imageWidth = 640;                             //����ͷ�ķֱ���  
const int imageHeight = 480;
Vec3f  point3;
float d;
Size imageSize = Size(imageWidth, imageHeight);

Mat img;
Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;
Rect m_l_select;
Rect m_r_select;

Rect validROIL;//ͼ��У��֮�󣬻��ͼ����вü��������validROI����ָ�ü�֮�������  
Rect validROIR;

Mat mapLx, mapLy, mapRx, mapRy;     //ӳ���  
Mat Rl, Rr, Pl, Pr, Q;              //У����ת����R��ͶӰ����P ��ͶӰ����Q
Mat xyz;              //��ά����

Point origin;         //��갴�µ���ʼ��
Rect selection;      //�������ѡ��
bool selectObject = false;    //�Ƿ�ѡ�����




int blockSize = 8, uniquenessRatio = 0, numDisparities = 3;
Ptr<StereoBM> bm = StereoBM::create(16, 9);

/*���ȱ궨�õ���������ڲξ���
fx 0 cx
0 fy cy
0  0  1
*/
Mat cameraMatrixL = (Mat_<double>(3, 3) << 516.5066236, -1.444673028, 320.2950423, 0, 516.5816117, 270.7881873, 0, 0, 1.);
//��õĻ������



/*418.523322187048	0	0
-1.26842201390676	421.222568242056	0
344.758267538961	243.318992284899	1 */ //2

Mat distCoeffL = (Mat_<double>(5, 1) << -0.046645194, 0.077595167, 0.012476819, -0.000711358, 0);
//[0.006636837611004,0.050240447649195] [0.006681263320267,0.003130367429418]


/*���ȱ궨�õ���������ڲξ���
fx 0 cx
0 fy cy
0  0  1
*/
Mat cameraMatrixR = (Mat_<double>(3, 3) << 511.8428182, 1.295112628, 317.310253, 0, 513.0748795, 269.5885026, 0, 0, 1);


/*
417.417985082506	0	0
0.498638151824367	419.795432389420	0
309.903372309072	236.256106972796	1
*/ //2


Mat distCoeffR = (Mat_<double>(5, 1) << -0.061588946, 0.122384376, 0.011081232, -0.000750439, 0);
//[-0.038407383078874,0.236392800301615]  [0.004121779274885,0.002296129959664]



Mat T = (Mat_<double>(3, 1) << -120.3559901, -0.188953775, -0.662073075);//Tƽ������
//[-1.210187345641146e+02,0.519235426836325,-0.425535566316217]
//��ӦMatlab����T����
//Mat rec = (Mat_<double>(3, 1) << -0.00306, -0.03207, 0.00206);//rec��ת��������Ӧmatlab om����  �� 
Mat rec = (Mat_<double>(3, 3) << 0.999911333, -0.004351508, 0.012585312,
	0.004184066, 0.999902792, 0.013300386,
	-0.012641965, -0.013246549, 0.999832341);                //rec��ת��������Ӧmatlab om����  �� 

/* 0.999341122700880	0.000660748031451783	-0.0362888948713456
-0.00206388651740061	0.999250989651683	-0.0386419468010579
0.0362361815232777	0.0386913826603732	0.998593969567432 */

//Mat T = (Mat_<double>(3, 1) << -48.4, 0.241, -0.0344);//Tƽ������
//[-1.210187345641146e+02,0.519235426836325,-0.425535566316217]
//��ӦMatlab����T����

Mat R;//R ��ת����


/*****����ƥ��*****/
void stereo_match(int, void*)
{
	bm->setBlockSize(2 * blockSize + 5);     //SAD���ڴ�С��5~21֮��Ϊ��
	bm->setROI1(validROIL);
	bm->setROI2(validROIR);
	bm->setPreFilterCap(31);
	bm->setMinDisparity(0);  //��С�ӲĬ��ֵΪ0, �����Ǹ�ֵ��int��
	bm->setNumDisparities(numDisparities * 16 + 16);//�Ӳ�ڣ�������Ӳ�ֵ����С�Ӳ�ֵ֮��,���ڴ�С������16����������int��
	bm->setTextureThreshold(10);
	bm->setUniquenessRatio(uniquenessRatio);//uniquenessRatio��Ҫ���Է�ֹ��ƥ��
	bm->setSpeckleWindowSize(100);
	bm->setSpeckleRange(32);
	bm->setDisp12MaxDiff(-1);
	Mat disp, disp8;
	bm->compute(rectifyImageL, rectifyImageR, disp);//����ͼ�����Ϊ�Ҷ�ͼ
	disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16) * 16.));//��������Ӳ���CV_16S��ʽ
	reprojectImageTo3D(disp, xyz, Q, true); //��ʵ�������ʱ��ReprojectTo3D������X / W, Y / W, Z / W��Ҫ����16(Ҳ����W����16)�����ܵõ���ȷ����ά������Ϣ��
	xyz = xyz * 16;
	imshow("disparity", disp8);
}

/*****�������������ص�*****/
static void onMouse(int event, int x, int y, int, void*)
{
	if (selectObject)
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = std::abs(x - origin.x);
		selection.height = std::abs(y - origin.y);
	}

	switch (event)
	{
	case EVENT_LBUTTONDOWN:   //�����ť���µ��¼�
		origin = Point(x, y);
		selection = Rect(x, y, 0, 0);
		selectObject = true;
		//cout << origin << "in world coordinate is: " << xyz.at<Vec3f>(origin) << endl;
		point3 = xyz.at<Vec3f>(origin);
		point3[0];
		//cout << "point3[0]:" << point3[0] << "point3[1]:" << point3[1] << "point3[2]:" << point3[2]<<endl;
		cout << "�������꣺" << endl;
		cout << "x: " << point3[0] << "  y: " << point3[1] << "  z: " << point3[2] << endl;
		d = point3[0] * point3[0] + point3[1] * point3[1] + point3[2] * point3[2];
		d = sqrt(d);   //mm
		// cout << "������:" << d << "mm" << endl;

		d = d / 10.0;   //cm
		cout << "������:" << d << "cm" << endl;

		// d = d/1000.0;   //m
		// cout << "������:" << d << "m" << endl;

		break;
	case EVENT_LBUTTONUP:    //�����ť�ͷŵ��¼�
		selectObject = false;
		if (selection.width > 0 && selection.height > 0)
			break;
	}
}


/*****������*****/
int test2()
{
	/*
	����У��
	*/
	Rodrigues(rec, R); //Rodrigues�任
	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
		0, imageSize, &validROIL, &validROIR);
	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

	/*
	��ȡͼƬ
	*/

	m_l_select = Rect(0, 0, 640, 480);
	img = imread("car.jpg", IMREAD_COLOR);
	//imshow("Image", img);
	rgbImageL = img(m_l_select);
	cvtColor(rgbImageL, grayImageL, COLOR_BGR2GRAY);

	m_r_select = Rect(640, 0, 640, 480);
	rgbImageR = img(m_r_select);
	cvtColor(rgbImageR, grayImageR, COLOR_BGR2GRAY);

	//imshow("ImageL", rgbImageL);
	//imshow("ImageR", rgbImageR);

	/*
	����remap֮�����������ͼ���Ѿ����沢���ж�׼��
	*/
	remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
	remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

	/*
	��У�������ʾ����
	*/
	Mat rgbRectifyImageL, rgbRectifyImageR;
	cvtColor(rectifyImageL, rgbRectifyImageL, COLOR_GRAY2BGR);  //α��ɫͼ
	cvtColor(rectifyImageR, rgbRectifyImageR, COLOR_GRAY2BGR);

	//������ʾ
	//rectangle(rgbRectifyImageL, validROIL, Scalar(0, 0, 255), 3, 8);
	//rectangle(rgbRectifyImageR, validROIR, Scalar(0, 0, 255), 3, 8);
	//imshow("ImageL After Rectify", rgbRectifyImageL);
	//imshow("ImageR After Rectify", rgbRectifyImageR);

	/*
	����ƥ��
	*/
	namedWindow("disparity", WINDOW_AUTOSIZE);

	/*************************���ο��ӻ�**********************************************/
	// ����SAD���� Trackbar
	//createTrackbar("BlockSize:\n", "disparity", &blockSize, 8, stereo_match);
	// �����Ӳ�Ψһ�԰ٷֱȴ��� Trackbar
	//createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, stereo_match);
	// �����Ӳ�� Trackbar
	//createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, stereo_match);


	//�����Ӧ����setMouseCallback(��������, ���ص�����, �����ص������Ĳ�����һ��ȡ0)
	setMouseCallback("disparity", onMouse, 0);
	stereo_match(0, 0);

	waitKey();
	return 0;
}












