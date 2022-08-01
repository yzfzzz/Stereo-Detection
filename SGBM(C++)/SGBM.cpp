/*        双目测距        */

#include <opencv2/opencv.hpp>  
#include <iostream>  
#include <math.h> 

using namespace std;
using namespace cv;

const int imageWidth = 640;                             //摄像头的分辨率  
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

Rect validROIL;//图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  
Rect validROIR;

Mat mapLx, mapLy, mapRx, mapRy;     //映射表  
Mat Rl, Rr, Pl, Pr, Q;              //校正旋转矩阵R，投影矩阵P 重投影矩阵Q
Mat xyz;              //三维坐标

Point origin;         //鼠标按下的起始点
Rect selection;      //定义矩形选框
bool selectObject = false;    //是否选择对象

int blockSize = 8, mindisparity = 1, ndisparities = 64, img_channels = 3;
Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(mindisparity, ndisparities, blockSize);



/*事先标定好的左相机的内参矩阵
fx 0 cx
0 fy cy
0  0  1
*/
Mat cameraMatrixL = (Mat_<double>(3, 3) << 516.5066236, -1.444673028, 320.2950423, 0, 516.5816117, 270.7881873, 0, 0, 1.);
//获得的畸变参数



/*418.523322187048	0	0
-1.26842201390676	421.222568242056	0
344.758267538961	243.318992284899	1 */ //2

Mat distCoeffL = (Mat_<double>(5, 1) << -0.046645194, 0.077595167, 0.012476819, -0.000711358, 0);
//[0.006636837611004,0.050240447649195] [0.006681263320267,0.003130367429418]


/*事先标定好的右相机的内参矩阵
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



Mat T = (Mat_<double>(3, 1) << -120.3559901, -0.188953775, -0.662073075);//T平移向量
//[-1.210187345641146e+02,0.519235426836325,-0.425535566316217]
															 //对应Matlab所得T参数
//Mat rec = (Mat_<double>(3, 1) << -0.00306, -0.03207, 0.00206);//rec旋转向量，对应matlab om参数  我 
Mat rec = (Mat_<double>(3, 3) << 0.999911333, -0.004351508, 0.012585312,
	0.004184066, 0.999902792, 0.013300386,
	-0.012641965, -0.013246549, 0.999832341);                //rec旋转向量，对应matlab om参数  我 

/* 0.999341122700880	0.000660748031451783	-0.0362888948713456
-0.00206388651740061	0.999250989651683	-0.0386419468010579
0.0362361815232777	0.0386913826603732	0.998593969567432 */

//Mat T = (Mat_<double>(3, 1) << -48.4, 0.241, -0.0344);//T平移向量
																							  //[-1.210187345641146e+02,0.519235426836325,-0.425535566316217]
																							  //对应Matlab所得T参数

Mat R;//R 旋转矩阵


	  /*****立体匹配*****/
void stereo_match(int, void*)
{
	/*
	bm->setBlockSize(2 * blockSize + 5);     //SAD窗口大小，5~21之间为宜
	bm->setROI1(validROIL);
	bm->setROI2(validROIR);
	bm->setPreFilterCap(31);
	bm->setMinDisparity(0);  //最小视差，默认值为0, 可以是负值，int型
	bm->setNumDisparities(numDisparities * 16 + 16);//视差窗口，即最大视差值与最小视差值之差,窗口大小必须是16的整数倍，int型
	bm->setTextureThreshold(10);
	bm->setUniquenessRatio(uniquenessRatio);//uniquenessRatio主要可以防止误匹配
	bm->setSpeckleWindowSize(100);
	bm->setSpeckleRange(32);
	bm->setDisp12MaxDiff(-1);
	*/

	int P1 = 8 * img_channels * blockSize * blockSize;
	int P2 = 32 * img_channels * blockSize * blockSize;
	sgbm->setP1(P1);
	sgbm->setP2(P2);
	sgbm->setPreFilterCap(1);
	sgbm->setUniquenessRatio(10);
	sgbm->setSpeckleRange(100);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setDisp12MaxDiff(-1);
	//sgbm->setNumDisparities(1);
	sgbm->setMode(cv::StereoSGBM::MODE_HH);
	
	Mat disp, disp8;
	sgbm->compute(rectifyImageL, rectifyImageR, disp);//输入图像必须为灰度图
	disp8 = Mat(disp.rows, disp.cols, CV_8UC1);
	normalize(disp, disp8, 0, 255, NORM_MINMAX, CV_8UC1);
	reprojectImageTo3D(disp, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
	xyz = xyz * 16;
	imshow("disparity", disp8);
}

/*****描述：鼠标操作回调*****/
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
	case EVENT_LBUTTONDOWN:   //鼠标左按钮按下的事件
		origin = Point(x, y);
		selection = Rect(x, y, 0, 0);
		selectObject = true;
		//cout << origin << "in world coordinate is: " << xyz.at<Vec3f>(origin) << endl;
		point3 = xyz.at<Vec3f>(origin);
		point3[0];
		//cout << "point3[0]:" << point3[0] << "point3[1]:" << point3[1] << "point3[2]:" << point3[2]<<endl;
		cout << "世界坐标：" << endl;
		cout << "x: " << point3[0] << "  y: " << point3[1] << "  z: " << point3[2] << endl;
		d = point3[0] * point3[0] + point3[1] * point3[1] + point3[2] * point3[2];
		d = sqrt(d);   //mm
	   // cout << "距离是:" << d << "mm" << endl;

		d = d / 10.0;   //cm
		cout << "距离是:" << d << "cm" << endl;

		// d = d/1000.0;   //m
		// cout << "距离是:" << d << "m" << endl;

		break;
	case EVENT_LBUTTONUP:    //鼠标左按钮释放的事件
		selectObject = false;
		if (selection.width > 0 && selection.height > 0)
			break;
	}
}


/*****主函数*****/
int main()
{
	/*
	立体校正
	*/
	Rodrigues(rec, R); //Rodrigues变换
	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
		0, imageSize, &validROIL, &validROIR);
	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

	/*
	读取图片
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
	经过remap之后，左右相机的图像已经共面并且行对准了
	*/
	remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
	remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

	/*
	把校正结果显示出来
	*/
	Mat rgbRectifyImageL, rgbRectifyImageR;
	cvtColor(rectifyImageL, rgbRectifyImageL, COLOR_GRAY2BGR);  //伪彩色图
	cvtColor(rectifyImageR, rgbRectifyImageR, COLOR_GRAY2BGR);

	//单独显示
	//rectangle(rgbRectifyImageL, validROIL, Scalar(0, 0, 255), 3, 8);
	//rectangle(rgbRectifyImageR, validROIR, Scalar(0, 0, 255), 3, 8);
	//imshow("ImageL After Rectify", rgbRectifyImageL);
	//imshow("ImageR After Rectify", rgbRectifyImageR);

	//显示在同一张图上
	Mat canvas;
	double sf;
	int w, h;
	sf = 600. / MAX(imageSize.width, imageSize.height);
	w = cvRound(imageSize.width * sf);
	h = cvRound(imageSize.height * sf);
	canvas.create(h, w * 2, CV_8UC3);   //注意通道

										//左图像画到画布上
	Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //得到画布的一部分  
	resize(rgbRectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //把图像缩放到跟canvasPart一样大小  
	Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //获得被截取的区域    
		cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
	//rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //画上一个矩形  
	cout << "Painted ImageL" << endl;

	//右图像画到画布上
	canvasPart = canvas(Rect(w, 0, w, h));                                      //获得画布的另一部分  
	resize(rgbRectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
	Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
		cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
	//rectangle(canvasPart, vroiR, Scalar(0, 0, 255), 3, 8);
	cout << "Painted ImageR" << endl;

	//画上对应的线条
	for (int i = 0; i < canvas.rows; i += 16)
		line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
	imshow("rectified", canvas);

	/*
	立体匹配
	*/
	namedWindow("disparity", WINDOW_AUTOSIZE);

	/*************************调参可视化**********************************************/
	// 创建SAD窗口 Trackbar
	//createTrackbar("BlockSize:\n", "disparity", &blockSize, 8, stereo_match);
	// 创建视差唯一性百分比窗口 Trackbar
	//createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, stereo_match);
	// 创建视差窗口 Trackbar
	//createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, stereo_match);


	//鼠标响应函数setMouseCallback(窗口名称, 鼠标回调函数, 传给回调函数的参数，一般取0)
	setMouseCallback("disparity", onMouse, 0);
	stereo_match(0, 0);

	waitKey();
	return 0;
}