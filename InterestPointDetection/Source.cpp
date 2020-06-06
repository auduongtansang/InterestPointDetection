#include <opencv2/opencv.hpp>
#include "InterestPointDetection.h"

using namespace cv;

int main()
{
	Mat src1 = imread("box.png", IMREAD_GRAYSCALE), src2 = imread("box_in_scene.png", IMREAD_GRAYSCALE);
	Mat dst;

	InterestPointDetection detector;
	detector.matchBySIFT(src1, src2, dst);

	namedWindow("Result");
	imshow("Result", dst);

	waitKey(0);
	destroyAllWindows();

	return 0;
}
