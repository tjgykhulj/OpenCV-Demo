#include <opencv2\opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
	//读入图，转为黑白
	const char *filename = "test.png";
	Mat src = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	uchar *p = src.data;
	int size = src.rows * src.cols;
	double s[256] = { 0 };
	for (int i = 0; i < size; i++) s[p[i]]++;	//统计直方图
	for (int i = 1; i < 256; i++) s[i] += s[i - 1];	//累计直方图
	for (int i = 0; i < 256; i++) s[i] = (s[i] / size) * 255;		//归一化累计分布取整
	for (int i = 0; i < size; i++) p[i] = (int) s[p[i]];	//映射
	imshow("Result", src);
	cvWaitKey(-1);
	cvDestroyAllWindows();
}