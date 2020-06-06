#pragma once

#include <opencv2/opencv.hpp>
#include "Convolution.h"
#include <vector>

using namespace cv;
using namespace std;

typedef vector<int> keypoint;

class InterestPointDetection
{
public:
	int detectHarris(const Mat& src, Mat& dst, double coef, double th);

	int detectBlob(const Mat& src, Mat& dst, double sigma, double coef, double th);

	int detectDOG(const Mat& src, Mat& dst, vector<keypoint> keypoints, double sigma, double coef, double cth, double eth);

	InterestPointDetection();
	~InterestPointDetection();
};
