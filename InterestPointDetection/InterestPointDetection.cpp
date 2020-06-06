#include "InterestPointDetection.h"

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

				//Phân ngưỡng cực trị
				if ((isMaximum || isMinimum) && value > th)
					circle(dst, Point(j, i), sqrt2sig[k], Scalar(0, 0, 255));
			}
	}

	return 1;
}

int InterestPointDetection::detectDOG(const Mat& src, Mat& dst, vector<keypoint>& keypoints, double sigma, double coef, double cth, double eth)
{
	//Nếu ảnh input rỗng => không làm gì hết
	if (src.empty())
		return -1;

	int row = src.rows, col = src.cols;

	//Chuẩn hóa [0, 1]
	Mat norm;
	src.convertTo(norm, CV_64FC1, 1.0 / 255);

	//Tạo các giá trị sigma cho mỗi tỉ lệ tại mỗi octave
	double sig[5][5];
	sig[0][0] = sigma;

	for (int k = 1; k < 5; k++)
		sig[0][k] = sig[0][k - 1] * coef;

	double coef2 = coef * coef;
	for (int oct = 1; oct < 5; oct++)
		for (int k = 0; k < 5; k++)
			sig[oct][k] = sig[oct - 1][k] * coef2;

	//Tạo không gian tỉ lệ Gaussian
	Mat Gaussian[5][5];
	int scaleRow = row * 2, scaleCol = col * 2;

	for (int oct = 0; oct < 5; oct++)
	{
		Mat scale;
		resize(norm, scale, Size(scaleCol, scaleRow));

		for (int k = 0; k < 5; k++)
			GaussianBlur(scale, Gaussian[oct][k], Size(0, 0), sig[oct][k]);

		scaleRow = int(round(scaleRow * 0.5));
		scaleCol = int(round(scaleCol * 0.5));
	}

	//Tạo không gian tỉ lệ DOG
	Mat DOG[5][4];

	for (int oct = 0; oct < 5; oct++)
		for (int k = 0; k < 4; k++)
		{
			Mat temp = Gaussian[oct][k + 1] - Gaussian[oct][k];

			pow(temp, 2, DOG[oct][k]);
		}

	//Định vị keypoint và hiển thị lên ảnh kết quả
	keypoints.resize(0);
	cvtColor(src, dst, COLOR_GRAY2BGR);

	scaleRow = row * 2;
	scaleCol = col * 2;

	double sqrt2 = sqrt(2);
	double rescale = 0.5;

	for (int oct = 0; oct < 5; oct++)
	{
		for (int k = 1; k < 3; k++)
		{
			double* DOGkdata = (double*)(DOG[oct][k].data);

			for (int i = 1; i < scaleRow - 1; i++)
				for (int j = 1; j < scaleCol - 1; j++)
				{
					int center = i * scaleCol + j;

					double value = *(DOGkdata + center);
					bool isMaximum = true, isMinimum = true;

					for (int t = -1; t <= 1; t++)
					{
						double* DOGtdata = (double*)(DOG[oct][k + t].data);

						for (int u = -1; u <= 1; u++)
							for (int v = -1; v <= 1; v++)
								if (t != 0 || u != 0 || v != 0)
								{
									int cur = center + u * scaleCol + v;
									double neighbor = *(DOGtdata + cur);

									if (value < neighbor)
										isMaximum = false;

									if (value > neighbor)
										isMinimum = false;
								}
					}

					//Phân ngưỡng cực trị
					if (isMaximum || isMinimum)
					{
						//Khai triển Taylor xung quanh cực trị ứng viên
						double* DOGadata = (double*)(DOG[oct][k + 1].data);
						double* DOGbdata = (double*)(DOG[oct][k - 1].data);

						//Đạo hàm bậc 1 của hàm DOG(x, y, sig)
						Mat DX = Mat(3, 1, CV_64FC1);
						double* DXdata = (double*)(DX.data);

						*(DXdata) = *(DOGkdata + center + 1) - *(DOGkdata + center - 1);
						*(DXdata + 1) = *(DOGkdata + center + scaleCol) - *(DOGkdata + center - scaleCol);
						*(DXdata + 2) = *(DOGadata + center) - *(DOGbdata + center);

						//Đạo hàm bậc 2 của hàm DOG(x, y, sig)
						Mat DXX = Mat(3, 3, CV_64FC1);
						double* DXXdata = (double*)(DXX.data);

						*(DXXdata) = *(DOGkdata + center + 1) + *(DOGkdata + center - 1) - 2 * *(DOGkdata + center);
						*(DXXdata + 4) = *(DOGkdata + center + scaleCol) + *(DOGkdata + center - scaleCol) - 2 * *(DOGkdata + center);
						*(DXXdata + 8) = *(DOGadata + center) + *(DOGbdata + center) - 2 * *(DOGkdata + center);

						*(DXXdata + 3) = *(DOGkdata + center + scaleCol + 1) + *(DOGkdata + center - scaleCol - 1);
						*(DXXdata + 3) -= *(DOGkdata + center - scaleCol + 1) + *(DOGkdata + center + scaleCol - 1);
						*(DXXdata + 1) = *(DXXdata + 3);

						*(DXXdata + 6) = *(DOGadata + center + 1) + *(DOGadata + center - 1);
						*(DXXdata + 6) -= *(DOGbdata + center + 1) + *(DOGbdata + center - 1);
						*(DXXdata + 2) = *(DXXdata + 6);

						*(DXXdata + 7) = *(DOGadata + center + scaleCol) + *(DOGadata + center - scaleCol);
						*(DXXdata + 7) -= *(DOGbdata + center + scaleCol) + *(DOGbdata + center - scaleCol);
						*(DXXdata + 5) = *(DXXdata + 7);

						//Điểm cực trị của hàm DOG(x, y, sig) xung quanh cực trị ứng viên
						Mat X = -DXX.inv() * DX; 

						//Giá trị cực trị
						Mat expr = 0.5 * DX.t() * X;
						double extrema = value + *(double*)(expr.data);

						//Chỉ giữ lại cực trị có độ tương phản cao
						if (extrema > cth)
						{
							//Xấp xỉ tỉ lệ hai trị riêng
							double eRatio = (*(DXXdata) + *(DXXdata + 4)) * (*(DXXdata) + *(DXXdata + 4));
							eRatio /= *(DXXdata) * *(DXXdata + 4) - *(DXXdata + 1) * *(DXXdata + 1);

							//Chỉ giữ lại cực trị không nằm trên biên cạnh
							if (eRatio < (eth + 1) * (eth + 1) / eth)
							{
								int originalX = int(round(j * rescale));
								int originalY = int(round(i * rescale));

								keypoint key{ double(oct), sig[oct][k], double(i), double(j) };
								keypoints.push_back(key);

								circle(dst, Point(originalX, originalY), 1, Scalar(0, 0, 255), -1);
							}
						}
					}

				}
		}

		scaleRow = int(round(scaleRow * 0.5));
		scaleCol = int(round(scaleCol * 0.5));
		rescale *= 2;
	}

	return 1;
}

int InterestPointDetection::extractSIFT(const Mat& src, const vector<keypoint>& keypoints, vector<descriptor>& descriptors)
{
	//Nếu ảnh input rỗng => không làm gì hết
	if (src.empty())
		return -1;

	int num = int(keypoints.size());
	descriptors.resize(0);

	for (int key = 0; key < num; key++)
	{
		int oct = int(round(keypoints[key][0]));
		double sig = keypoints[key][1];
		int y = int(round(keypoints[key][2]));
		int x = int(round(keypoints[key][3]));

		//Tạo ảnh Gaussian tại tỉ lệ cực đại của keypoint
		Mat scale, Gaussian;
		resize(src, scale, Size(int(round(src.cols * pow(2, -oct + 1))), int(round(src.rows * pow(2, -oct + 1)))));
		GaussianBlur(scale, Gaussian, Size(0, 0), sig);

		//Lấy vùng ảnh xung quanh keypoint
		int minRow = max(0, y - 8);
		int maxRow = min(Gaussian.rows - 1, y + 8);
		int minCol = max(0, x - 8);
		int maxCol = min(Gaussian.cols - 1, x + 8);
		
		Mat win = Gaussian(Range(minRow, maxRow), Range(minCol, maxCol));
		int wrow = win.rows, wcol = win.cols;

		//Tạo ma trận trọng số
		Mat weights = getGaussianKernel(wrow, 1.5 * sig) * getGaussianKernel(wcol, 1.5 * sig).t();

		//Tính đạo hàm theo phương
		Mat dX, dY;

		Convolution convolution;
		double ker[] { -1, 0, 1, };

		convolution.SetKernel(ker, 3, 1);
		convolution.DoConvolution(win, dX);

		convolution.SetKernel(ker, 1, 3);
		convolution.DoConvolution(win, dY);

		//Tính đạo hàm tổng hợp và hướng
		Mat mag = Mat(wrow, wcol, CV_64FC1, Scalar(0));
		Mat the = Mat(wrow, wcol, CV_64FC1, Scalar(0));

		double* dXdata = (double*)(dX.data);
		double* dYdata = (double*)(dY.data);

		double* magdata = (double*)(mag.data);
		double* thedata = (double*)(the.data);

		for (int i = 0; i < wrow; i++)
			for (int j = 0; j < wcol; j++)
			{
				int center = i * wcol + j;

				double dx = *(dXdata + center);
				double dy = *(dYdata + center);

				*(magdata + center) = sqrt(dx * dx + dy * dy);

				double angle = 0;
				if (dx == 0)
				{
					if (dy > 0)
						angle = 90;
					else if (dy < 0)
						angle = 270;
				}
				else
				{
					angle = atan(dy / dx) * 180 / M_PI;

					if (dx < 0)
						angle += 180;

					if (angle < 0)
						angle += 360;
				}
				*(thedata + center) = angle;
			}

		//Làm trơn giá trị đạo hàm tổng hợp
		mag = mag.mul(weights);

		//Chia 36 bin hướng, tính histogram
		vector<double> bin36(36, 0);

		for (int i = 0; i < wrow; i++)
			for (int j = 0; j < wcol; j++)
			{
				int center = i * wcol + j;

				int bin = int(*(thedata + center) / 10);

				bin36[bin] += *(magdata + center);
			}

		//Tìm hướng chính, gán cho keypoint
		double maximum = -999;
		int maxBin = -1;

		for (int i = 0; i < 36; i++)
			if (maximum < bin36[i])
			{
				maximum = bin36[i];
				maxBin = i;
			}

		//Xoay hướng keypoint về 0, tạo descriptor
		the = the - maxBin;

		descriptor des(128, 0);

		for (int i = 0; i < wrow; i++)
			for (int j = 0; j < wcol; j++)
			{
				int center = i * wcol + j;

				double angle = *(thedata + center);
				if (angle < 0)
					angle += 360;

				int bin = int(angle / 45);

				//Vị trí của bin này trong des
				int pos = (int(i / 4) * 4 + int(j / 4)) * 8 + bin;

				des[pos] += *(magdata + center);
			}

		//Chuẩn hóa
		double length = 0;

		for (int i = 0; i < 128; i++)
			length += des[i] * des[i];

		length = sqrt(length);

		for (int i = 0; i < 128; i++)
			des[i] /= length;

		descriptors.push_back(des);
	}

	return 1;
}

int InterestPointDetection::matchBySIFT(const Mat& src1, const Mat& src2, Mat& dst)
{
	//Nếu ảnh input rỗng => không làm gì hết
	if (src1.empty() || src2.empty())
		return -1;

	Mat dst1, dst2;
	vector<keypoint> key1, key2;
	vector<descriptor> des1, des2;

	//Tìm keypoint
	detectDOG(src1, dst1, key1, 1, sqrt(2), 0.005, 20);
	detectDOG(src2, dst2, key2, 1, sqrt(2), 0.007, 20);

	//Tạo descriptor cho mỗi keypoint
	extractSIFT(src1, key1, des1);
	extractSIFT(src2, key2, des2);

	//Hiển thị ảnh kết quả
	dst = Mat(dst2.rows, dst1.cols + dst2.cols, CV_8UC3, Scalar(0));
	dst1.copyTo(dst(Rect(0, 0, dst1.cols, dst1.rows)));
	dst2.copyTo(dst(Rect(dst1.cols, 0, dst2.cols, dst2.rows)));

	//So khớp descriptor bằng 1-NN
	for (int i = 0; i < des1.size(); i++)
	{
		double min = 9999999;
		int argmin = -1;

		for (int j = 0; j < des2.size(); j++)
		{
			double sdist = 0;
			for (int k = 0; k < 128; k++)
				sdist += pow(des1[i][k] - des2[j][k], 2);
			if (sdist < min)
			{
				min = sdist;
				argmin = j;
			}
		}

		//Phân ngưỡng khoảng cách
		if (min <= 0.1)
		{
			int x1 = int(round(key1[i][3] * pow(2, key1[i][0] - 1)));
			int y1 = int(round(key1[i][2] * pow(2, key1[i][0] - 1)));

			int x2 = int(round(key2[argmin][3] * pow(2, key2[argmin][0] - 1))) + dst1.cols;
			int y2 = int(round(key2[argmin][2] * pow(2, key2[argmin][0] - 1)));

			line(dst, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0));
		}
	}

	return 1;
}
