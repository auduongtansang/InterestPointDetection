#pragma once

#define _USE_MATH_DEFINES

#include <cmath>
#include <opencv2/opencv.hpp>

using namespace cv;

class Convolution
{

private:

	Mat _k;  //Ma trận kernel
	int _row;  //Số dòng kernel
	int _col;  //Số cột kernel

public:
	//Trả về kernel hiện tại
	Mat GetKernel();

	//Thiết lập kernel, số dòng và số cột phải là số lẻ
	void SetKernel(double k[], int row, int col);

	//Thiết lập kernel tỉ lệ chuẩn hóa LOG
	void SetScaleNormalizedLOG(double sigma);

	/*
	Hàm tích chập của một ảnh xám với kernel hiện tại
	src: ảnh input
	dst: ảnh output
	Hàm trả về:
		1: nếu tính thành công
		-1: nếu tính thất bại (không đọc được ảnh input,...)
	*/
	int DoConvolution(const Mat& src, Mat& dst);

	Convolution();
	~Convolution();
};
