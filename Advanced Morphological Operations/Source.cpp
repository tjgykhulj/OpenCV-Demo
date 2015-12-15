#include <opencv2\opencv.hpp>

#include "MyCV.h"
using namespace std;
using namespace cv;

int main()
{
	//����ڰ�ͼ��תΪMat���ҵ�threshold����ֵ����
	const char *filename = "test.png";
	Mat input0 = imread(filename);
	Mat input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	int rows = input.rows;
	int cols = input.cols;
	int size = input.cols * input.rows;
	int threshold1 = OstuThreshold(input);
	Mat skelet(rows, cols, input.type());
	Mat temp(rows, cols, input.type());
	uchar *o = skelet.data;
	for (int i = 0; i < size; i++) o[i] = (input.data[i] >= threshold1);
	imshow("Binary1", skelet * 255);
	Closing(skelet, default_str_element);
	imshow("Binary2", skelet * 255);
	//������
	while (Thinging(skelet, 8, ele));
	imshow("Thinging", skelet * 255);

	int threshold2 = FuzzyThreshold(input);
	for (int i = 0; i < size; i++) temp.data[i] = (input.data[i] >= threshold2);
	Closing(temp, default_str_element);

	for (int i = 0; i < rows-1; i++)
		for (int j = 0; j < cols; j++)
		{
			int k = i*cols + j;
			if (temp.data[k] && input.data[k] < threshold2)	//�˵�Ϊ��ɫ������closingǰΪ��
				input0.at<Vec3b>(i, j) = { 200, 0, 0 };
			else if (o[k]) 
				input0.at<Vec3b>(i, j) = { 0, 0, 200 };		//���˵�Ϊ�������ϵĵ�
		}
	
	Thickening(skelet, 8, ele);
	imshow("Thickening", skelet * 255);
	imshow("Result", input0);
	cvWaitKey(-1);
	cvDestroyAllWindows();
}