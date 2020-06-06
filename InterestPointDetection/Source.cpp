#include <opencv2/opencv.hpp>
#include "InterestPointDetection.h"

using namespace cv;

int main()
{
	Mat src = imread("butt.jpg", IMREAD_GRAYSCALE);
	Mat dst;

	namedWindow("Source");
	imshow("Source", src);

	InterestPointDetection detector;
	detector.detectDOG(src, dst, 1, sqrt(2), 0.005);

	namedWindow("Result");
	imshow("Result", dst);

	waitKey(0);
	destroyAllWindows();

	return 0;
}
