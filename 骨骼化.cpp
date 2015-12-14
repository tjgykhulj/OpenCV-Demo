#include <opencv2\opencv.hpp>

#include "MyCV.h"
using namespace std;
using namespace cv;

int main()
{
	//读入黑白图，转为Mat，找到threshold并二值化。
	Mat input = imread("test.png", CV_LOAD_IMAGE_GRAYSCALE);
	int rows = input.rows;
	int cols = input.cols;
	int size = input.cols * input.rows;
	int threshold = OstuThreshold(input);
	Mat output(rows, cols, input.type());
	Mat skelet(rows, cols, input.type());
	uchar *o = output.data;
	uchar *s = skelet.data;
	for (int i = 0; i < size; i++) o[i] = (input.data[i] >= threshold);
	imshow("binary0", output * 255);
		Dilation(output, default_str_element);
		Erosion(output, default_str_element);
	imshow("binary1", output * 255);

	int a[9];
	a[0] = -cols; a[1] = a[0] + 1; a[7] = a[0] - 1; a[6] = -1;
	a[4] = cols;  a[3] = a[4] + 1; a[5] = a[4] - 1; a[2] = 1;
	a[8] = a[0];
	/* 7 0 1
	 * 6 X 2	另外第8设为0方便计算
	 * 5 4 3
	 */

	for (;;) {
		for (int i = cols; i < size - cols; i++)
		{
			s[i] = o[i];
			if (!o[i]) continue;
			if (i % cols == 0 || i % cols == cols - 1) continue;
			int np = 0;
			for (int j = 0; j < 8; j++) np += o[i + a[j]];
			if (np < 2 || np > 6) continue;
			int sp = 0;
			for (int j = 0; j < 8; j++) sp += (!o[i + a[j]] && o[i + a[j + 1]]);
			if (sp != 1) continue;
			if (o[i + a[0]] && o[i + a[2]] && o[i + a[4]]) continue;
			if (o[i + a[6]] && o[i + a[2]] && o[i + a[4]]) continue;
			s[i] = 0;
		}
		for (int i = cols; i < size - cols; i++) if (o[i])
		{
			if (i % cols == 0 || i % cols == cols - 1) continue;
			int np = 0;
			for (int j = 0; j < 8; j++) np += o[i + a[j]];
			if (np < 2 || np > 6) continue;
			int sp = 0;
			for (int j = 0; j < 8; j++) sp += (!o[i + a[j]] && o[i + a[j + 1]]);
			if (sp != 1) continue;
			if (o[i + a[0]] && o[i + a[2]] && o[i + a[6]]) continue;
			if (o[i + a[0]] && o[i + a[4]] && o[i + a[6]]) continue;
			s[i] = 0;
		}
		for (int i = 0; i < size; i++) o[i] = s[i];
		imshow("Fuzzy", output * 255);
		cvWaitKey(-1);
	}
	cvDestroyAllWindows();
}