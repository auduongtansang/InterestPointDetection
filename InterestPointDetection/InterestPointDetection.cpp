﻿#include "InterestPointDetection.h"

InterestPointDetection::InterestPointDetection()
{
}

InterestPointDetection::~InterestPointDetection()
{
}

int InterestPointDetection::detectHarris(const Mat& src, Mat& dst, double coef, double th)
{
	//Nếu ảnh input rỗng => không làm gì hết
	if (src.empty())
		return -1;

	int row = src.rows, col = src.cols;

	//Làm trơn
	Mat blur;
	GaussianBlur(src, blur, Size(0, 0), 1);

	//Kernel đạo hàm Sobel
	double kX[] { -1, 0, 1, -2, 0, 2, -2, 0, 2 };
	double kY[] { -1, -2, -1, 0, 0, 0, 1, 2, 1 };

	//Tích chập
	Convolution convolution;
	Mat dX, dY;

	convolution.SetKernel(kX, 3, 3);
	convolution.DoConvolution(blur, dX);

	convolution.SetKernel(kY, 3, 3);
	convolution.DoConvolution(blur, dY);

	//Tính hệ số Harris cho mỗi điểm ảnh
	Mat harris = Mat(row, col, CV_64FC1, Scalar(0));
	double *harrisdata = (double*)(harris.data);

	double *dXdata = (double*)(dX.data), *dYdata = (double*)(dY.data);

	for (int i = 1; i < row - 1; i++)
		for (int j = 1; j < col - 1; j++)
		{
			int center = i * col + j;

			double dx = 0, dy = 0, dxdy = 0;

			for (int u = -1; u <= 1; u++)
				for (int v = -1; v <= 1; v++)
				{
					int cur = center + u * col + v;

					double ix = *(dXdata + cur);
					double iy = *(dYdata + cur);

					dx += ix * ix;
					dy += iy * iy;
					dxdy += ix * iy;
				}

			*(harrisdata + center) = dx * dy - dxdy * dxdy - coef * (dx + dy) * (dx + dy);
		}

	//Lọc cực trị cục bộ
	double hMax = 0;

	for (int i = 1; i < row - 1; i++)
		for (int j = 1; j < col - 1; j++)
		{
			int center = i * col + j;

			double value = *(harrisdata + center);
			bool isMaximum = true, isMinimum = true;

			for (int u = -1; u <= 1; u++)
				for (int v = -1; v <= 1; v++)
					if (u != 0 || v != 0)
					{
						int cur = center + u * col + v;

						double neighbor = *(harrisdata + cur);

						if (value < neighbor)
							isMaximum = false;

						if (value > neighbor)
							isMinimum = false;
					}

			if (isMaximum || isMinimum)
				hMax = MAX(hMax, value);
			else
				*(harrisdata + center) = 0;
		}

	//Lọc ngưỡng và hiển thị lên ảnh kết quả
	cvtColor(src, dst, COLOR_GRAY2BGR);
	double thVal = th * hMax;

	for (int i = 1; i < row - 1; i++)
		for (int j = 1; j < col - 1; j++)
		{
			int center = i * col + j;

			if (*(harrisdata + center) > thVal)
				circle(dst, Point(j, i), 3, Scalar(0, 0, 255), -1);
		}

	return 1;
}

int InterestPointDetection::detectBlob(const Mat& src, Mat& dst, double sigma, double coef, double th)
{
	//Nếu ảnh input rỗng => không làm gì hết
	if (src.empty())
		return -1;

	int row = src.rows, col = src.cols;

	//Làm trơn
	Mat blur;
	GaussianBlur(src, blur, Size(0, 0), 1);

	//Chuẩn hóa [0, 1]
	Mat norm;
	blur.convertTo(norm, CV_64FC1, 1.0 / 255);

	//Tạo các giá trị sigma và radius tương ứng cho các tỉ lệ
	double sqrt2 = sqrt(2);
	double sig[10];
	short sqrt2sig[10];

	sig[0] = sigma;
	sqrt2sig[0] = short(ceil(sigma * sqrt2));

	for (int k = 1; k < 10; k++)
	{
		sig[k] = sig[k - 1] * coef;
		sqrt2sig[k] = short(ceil(sig[k] * sqrt2));
	}

	//Tạo không gian tỉ lệ chuẩn hóa LOG
	Convolution convolution;
	Mat LOG[10];

	for (int k = 0; k < 10; k++)
	{
		Mat temp;

		convolution.SetScaleNormalizedLOG(sig[k]);
		convolution.DoConvolution(norm, temp);

		pow(temp, 2, LOG[k]);
	}

	//Lọc cực trị cục bộ và hiển thị lên ảnh kết quả
	cvtColor(src, dst, COLOR_GRAY2BGR);

	for (int k = 1; k < 9; k++)
	{
		double* LOGkdata = (double*)(LOG[k].data);

		for (int i = 1; i < row - 1; i++)
			for (int j = 1; j < col - 1; j++)
			{
				int center = i * col + j;

				double value = *(LOGkdata + center);
				bool isMaximum = true, isMinimum = true;

				for (int t = -1; t <= 1; t++)
				{
					double* LOGtdata = (double*)(LOG[k + t].data);

					for (int u = -1; u <= 1; u++)
						for (int v = -1; v <= 1; v++)
							if (t != 0 || u != 0 || v != 0)
							{
								int cur = center + u * col + v;
								double neighbor = *(LOGtdata + cur);

								if (value < neighbor)
									isMaximum = false;

								if (value > neighbor)
									isMinimum = false;
							}
				}

				if ((isMaximum || isMinimum) && value > th)
					circle(dst, Point(j, i), sqrt2sig[k], Scalar(0, 0, 255));
			}
	}

	return 1;
}
