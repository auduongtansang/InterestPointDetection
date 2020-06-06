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

void Convolution::SetScaleNormalizedLOG(double sigma)
{
	_row = int(ceil(sigma * 6));

	if ((_row & 1) == 0)
		_row += 1;

	_col = _row;

	//Bán kính kernel
	int half = _row / 2;

	//Hằng số
	double sigma22 = sigma * sigma * 2;
	double pisiama42 = pow(sigma, 4) * M_PI * 2;

	_k = Mat(_row, _col, CV_64FC1, Scalar(0));
	double* data = (double*)(_k.data);

	for (int i = 0, ii = -half; i < _row; i++, ii++)
		for (int j = 0, jj = -half; j < _col; j++, jj++)
		{
			int center = i * _col + j;

			double ii2jj2 = (double)ii * ii + (double)jj * jj;
			*(data + center) = (ii2jj2 - sigma22) * exp(-ii2jj2 / sigma22) / pisiama42;
		}
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
