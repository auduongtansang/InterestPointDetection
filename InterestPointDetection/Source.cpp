#include <string>
#include <opencv2/opencv.hpp>
#include "InterestPointDetection.h"

using namespace cv;
using namespace std;

int main(int argc, char** args)
{
	if (argc < 5)
	{
		cout << "Khong du tham so!\n";
		return -1;
	}

	Mat src1, src2, dst;
	InterestPointDetection detector;

	if (strcmp(args[1], "harris") == 0)
	{
		src1 = imread(args[2], IMREAD_GRAYSCALE);

		if (src1.empty())
		{
			cout << "Khong the mo anh!\n";
			return -1;
		}

		namedWindow("Source");
		imshow("Source", src1);

		detector.detectHarris(src1, dst, stof(args[3]), stof(args[4]));

		namedWindow("Result");
		imshow("Result", dst);
	}
	else if (strcmp(args[1], "blob") == 0)
	{
		if (argc < 6)
		{
			cout << "Khong du tham so!\n";
			return -1;
		}

		src1 = imread(args[2], IMREAD_GRAYSCALE);

		if (src1.empty())
		{
			cout << "Khong the mo anh!\n";
			return -1;
		}

		namedWindow("Source");
		imshow("Source", src1);

		detector.detectBlob(src1, dst, stof(args[3]), stof(args[4]), stof(args[5]));

		namedWindow("Result");
		imshow("Result", dst);
	}
	else if (strcmp(args[1], "dog") == 0)
	{
		if (argc < 6)
		{
			cout << "Khong du tham so!\n";
			return -1;
		}

		src1 = imread(args[2], IMREAD_GRAYSCALE);

		if (src1.empty())
		{
			cout << "Khong the mo anh!\n";
			return -1;
		}

		namedWindow("Source");
		imshow("Source", src1);

		vector<keypoint> keys;
		detector.detectDOG(src1, dst, keys, stof(args[3]), stof(args[4]), stof(args[5]), stof(args[5]));

		namedWindow("Result");
		imshow("Result", dst);
	}
	else if (strcmp(args[1], "sift") == 0)
	{
		if (argc < 13)
		{
			cout << "Khong du tham so!\n";
			return -1;
		}

		src1 = imread(args[2], IMREAD_GRAYSCALE);
		src2 = imread(args[7], IMREAD_GRAYSCALE);

		if (src1.empty())
		{
			cout << "Khong the mo anh!\n";
			return -1;
		}

		namedWindow("Source1");
		imshow("Source1", src1);

		namedWindow("Source2");
		imshow("Source2", src2);

		vector<keypoint> keys;
		detector.matchBySIFT(src1, stof(args[3]), stof(args[4]), stof(args[5]), stof(args[6]), src2, stof(args[8]), stof(args[9]), stof(args[10]), stof(args[11]), stof(args[12]), dst);

		namedWindow("Result");
		imshow("Result", dst);
	}

	waitKey(0);
	destroyAllWindows();

	return 0;
}
