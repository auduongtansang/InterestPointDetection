#pragma once

#define _USE_MATH_DEFINES

#include <opencv2/opencv.hpp>
#include "Convolution.h"
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

typedef vector<double> keypoint;
typedef vector<double> descriptor;

class InterestPointDetection
{
public:
	int detectHarris(const Mat& src, Mat& dst, double coef, double th);

	int detectBlob(const Mat& src, Mat& dst, double sigma, double coef, double th);

	int detectDOG(const Mat& src, Mat& dst, vector<keypoint>& keypoints, double sigma, double coef, double cth, double eth);

	int extractSIFT(const Mat& src, const vector<keypoint>& keypoints, vector<descriptor>& descriptors);

	int matchBySIFT(const Mat& src1, double sigma1, double coef1, double cth1, double eth1, const Mat& src2, double sigma2, double coef2, double cth2, double eth2, double dth, Mat& dst);

	InterestPointDetection();
	~InterestPointDetection();
};
