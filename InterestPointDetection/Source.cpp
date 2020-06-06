#include <opencv2/opencv.hpp>
#include "InterestPointDetection.h"

using namespace cv;

int main()
{
	Mat src = imread("chess.jpg", IMREAD_GRAYSCALE);
	Mat dst;

	namedWindow("Source");
	imshow("Source", src);

	InterestPointDetection detector;
	detector.detectHarris(src, dst, 0.04, 0.001);

	namedWindow("Result");
	imshow("Result", dst);

	waitKey(0);
	destroyAllWindows();

	return 0;
}
