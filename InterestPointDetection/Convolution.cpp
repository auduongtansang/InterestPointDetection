#include "Convolution.h"

Convolution::Convolution()
{
	_row = _col = 0;
}

Convolution::~Convolution()
{
	_k = Mat();
	_row = _col = 0;
}

Mat Convolution::GetKernel()
{
	return _k.clone();
}

void Convolution::SetKernel(double k[], int row, int col)
{
	//Số dòng và số cột là số chẵn => không làm gì hết
	if ((row & 1) == 0 || (col & 1) == 0)
		return;

	_row = row;
	_col = col;
	
	_k = Mat(row, col, CV_64FC1, Scalar(0));
	double* data = (double*)(_k.data);

	for (int i = 0; i < row * col; i++)
		*(data + i) = k[i];
}

int Convolution::DoConvolution(const Mat& src, Mat& dst)
{
	//Nếu ảnh input rỗng => không làm gì hết
	if (src.empty())
		return -1;

	//Nếu kernel rỗng => không làm gì hết
	if (_row <= 0 || _col <= 0)
		return -1;

	Mat srccpy;
	src.convertTo(srccpy, CV_64FC1);
	filter2D(srccpy, dst, -1, _k);

	return 1;
}
