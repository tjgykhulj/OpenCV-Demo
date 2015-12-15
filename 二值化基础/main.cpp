#include <opencv2/opencv.hpp>
#pragma comment(linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"")

#define len 2

using namespace cv;

void dilation(const Mat &src, Mat &dst, const Mat &str_element)	//����
{
	Point anchor(str_element.cols / 2, str_element.rows / 2);
	dst = Mat(src.size(), src.type(), Scalar::all(0));
	for (int i = 0; i < str_element.rows; i++)
		for (int j = 0; j < str_element.cols; j++)
			if (str_element.at<uchar>(i, j))		//��ǰ��Ϊ1�Ĳ��ֲŻ�Ӱ�쵽��
				for (int k = 0; k < src.rows; k++)
					for (int t = 0; t < src.cols; t++)	
						if (src.at<uchar>(k, t))		//���˵�Ϊ1����ƫ��(i,j)-anchor���λ�õĵ���Ϊ1��ȡ�ϴ�ֵ����˼)
						{
							Point pos = Point(t + j, k + i) - anchor;
							if (pos.x >= 0 && pos.x < src.cols && pos.y >= 0 && pos.y < src.rows) dst.at<uchar>(pos) = 1;
						}
}

void erosion(const Mat &src, Mat &dst, const Mat &str_element)	//��ʴ
{
	Point anchor(str_element.cols / 2, str_element.rows / 2);
	dst = Mat(src.size(), src.type(), Scalar::all(1));
	for (int i = 0; i < str_element.rows; i++)
		for (int j = 0; j < str_element.cols; j++)
			if (str_element.at<uchar>(i, j))
				for (int k = 0; k < src.rows; k++)
					for (int t = 0; t < src.cols; t++)
						if (!src.at<uchar>(k, t))		//���˵�Ϊ0����ƫ��(i,j)-anchor���λ�õĵ���Ϊ0����ʵ��ȡ��Сֵ����˼)
						{
							Point pos = Point(t + j, k + i) - anchor;
							if (pos.x >= 0 && pos.x < src.cols && pos.y >= 0 && pos.y < src.rows) dst.at<uchar>(pos) = 0;
						}
}

int main()
{
	const char *filename = "sample.jpg";
	Mat src = Mat(cvLoadImage(filename, CV_LOAD_IMAGE_GRAYSCALE)) / 255;	//����ڰ�ͼ��/255����������ֵͼƬ��
	Mat dst_ero, dst_dil;
	Mat str_element = getStructuringElement(cv::MORPH_ELLIPSE, Size(2 * len + 1, 2 * len + 1));	//����structuring element
	//erode(src, dst_sample, str_element);
	erosion(src, dst_ero, str_element);
	dilation(src, dst_dil, str_element);	//������������

	imshow("src", src * 255);
	imshow("erosion", dst_ero * 255);
	imshow("dilation", dst_dil * 255);		//��ʾʱ����255��������ʾ��
	cvWaitKey();
	src.release();
	dst_ero.release();
	dst_dil.release();
	str_element.release();
	cvDestroyAllWindows();				//�ͷſռ�
}