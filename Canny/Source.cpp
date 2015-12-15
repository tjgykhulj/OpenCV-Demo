#include <opencv2\opencv.hpp>
using namespace std;
using namespace cv;

const int gSize = 5;
const float sigma = 1.5;
const int myCVType = CV_32F;
typedef float myType;
const int upThresh = 200;
const int lowThresh = upThresh * 0.4;

#define judge(x, y, value, mat) (x >= 0 && x < mat.cols && y >= 0 && y < mat.rows && value < mat.at<myType>(y, x))
static int a[8][2] = { { 1, 1 }, { 0, 1 }, { 1, -1 }, { 1, 0 }, { -1, -1 }, { 0, -1 }, { -1, 1 }, { -1, 0 } };

void followEdges(int x, int y, Mat &M, int lowThresh, Mat &ans)
{
	ans.at<myType>(y, x) = 1;
	for (int i = 0; i < 8; i++) {
		int xx = x + a[i][0];
		int yy = y + a[i][1];
		if (judge(xx, yy, lowThresh, M) && ans.at<myType>(yy, xx) != 1) followEdges(xx, yy, M, lowThresh, ans);
	}
}

void edgeDetect(Mat &M, int upThresh, int lowThresh, Mat &ans) 
{
	ans = Mat(M.size(), myCVType, 0.0);
	for (int x = 0; x < M.cols; x++)
	for (int y = 0; y < M.rows; y++)
		if (M.at<myType>(y, x) >= upThresh)
			followEdges(x, y, M, lowThresh, ans);
}

void NonMaximum(Mat &M, Mat &D) 
{
	for (int x = 0; x < M.cols; x++)
		for (int y = 0; y < M.rows; y++)
		{
			myType current = atan(D.at<myType>(y, x)) * (180 / CV_PI);
			while (current < 0) current += 180;
			D.at<myType>(y, x) = current;
			int k = (current - 22.5) / 45.0;
			int v = M.at<myType>(y, x);
			if (judge(x + a[k][0], y + a[k][1], v, M) ||
				judge(x - a[k][0], y - a[k][1], v, M)) M.at<myType>(y, x) = 0;
		}
}
int main()
{
	//¶ÁÈëÍ¼£¬×ªÎªºÚ°×
	const char *filename = "1.png";
	Mat src = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	int h = src.rows;
	int w = src.cols;

	Mat image(src.size(), src.type());
	GaussianBlur(src, image, Size(3, 3), 1.5);

	//calculate magnitudes and direction using Sobel
	Mat sobelX = Mat(src.rows, src.cols, myCVType);
	Mat sobelY = Mat(src.rows, src.cols, myCVType);
	Sobel(image, sobelX, myCVType, 1, 0, 3);
	Sobel(image, sobelY, myCVType, 0, 1, 3);
	//calculate slope
	Mat slopes = Mat(image.rows, image.cols, myCVType);
	Mat sum = Mat(image.rows, image.cols, myCVType);

	myType *s = (myType *) sum.data;
	myType *sl = (myType *)slopes.data;
	myType *sX = (myType *)sobelX.data;
	myType *sY = (myType *)sobelY.data;
	for (int i = 0; i < h*w; i++) {
		s[i] = sqrt(sX[i] * sX[i] + sY[i] * sY[i]);
		sl[i] = sY[i] / sX[i];
	}
	//Non Maximum Suppression
	NonMaximum(sum, slopes);
	//edge detection and following
	Mat ans;
	edgeDetect(sum, upThresh, lowThresh, ans);

	imshow("src", src);
	imshow("dst", sum / 256);
	imshow("ans", ans);
	cvWaitKey(-1);
	cvDestroyAllWindows();
}