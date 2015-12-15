#include <opencv2/opencv.hpp>
#include<opencv2/nonfree/nonfree.hpp>
using namespace std;
using namespace cv;

int main()
{
	//读入图，转为黑白
	Mat srcA = imread("1.jpg");
	Mat srcB = imread("2.png");
	//找到特征点
	SiftFeatureDetector detector;
	vector<KeyPoint> kpA, kpB;
	detector.detect(srcA, kpA);
	detector.detect(srcB, kpB);
	//提取特征向量
	SiftDescriptorExtractor extractor;
	Mat desA, desB;//descriptor  
	extractor.compute(srcA, kpA, desA);
	extractor.compute(srcB, kpB, desB);
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(desA, desB, matches);
	Mat img_match;
	drawMatches(srcA, kpA, srcB, kpB, matches, img_match);
	imshow("matches", img_match);

	cvWaitKey(-1);
	destroyAllWindows();
}