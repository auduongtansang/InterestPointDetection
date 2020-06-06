#pragma once

#include <opencv2/opencv.hpp>
#include "Convolution.h"

using namespace cv;

class InterestPointDetection
{
public:
	int detectHarris(const Mat& src, Mat& dst, double coef, double th);

	int detectBlob(const Mat& src, Mat& dst, double sigma, double coef, double th);

	InterestPointDetection();
	~InterestPointDetection();
};
