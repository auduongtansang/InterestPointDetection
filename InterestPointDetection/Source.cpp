#include <opencv2/opencv.hpp>
#include "InterestPointDetection.h"

using namespace cv;

int main()
{
	Mat src = imread("butt.jpg", IMREAD_GRAYSCALE);
	Mat dst;

	namedWindow("Source");
	imshow("Source", src);

	vector<keypoint> key;
	vector<descriptor> des;

	InterestPointDetection detector;
	detector.detectDOG(src, dst, key, 1, sqrt(2), 0.005, 10);
	detector.extractSIFT(src, key, des);

	namedWindow("Result");
	imshow("Result", dst);

	waitKey(0);
	destroyAllWindows();

	return 0;
}
