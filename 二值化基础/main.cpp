#include <opencv2/opencv.hpp>
#pragma comment(linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"")

#define len 2

using namespace cv;

void dilation(const Mat &src, Mat &dst, const Mat &str_element)	//膨胀
{
	Point anchor(str_element.cols / 2, str_element.rows / 2);
	dst = Mat(src.size(), src.type(), Scalar::all(0));
	for (int i = 0; i < str_element.rows; i++)
		for (int j = 0; j < str_element.cols; j++)
			if (str_element.at<uchar>(i, j))		//当前点为1的部分才会影响到它
				for (int k = 0; k < src.rows; k++)
					for (int t = 0; t < src.cols; t++)	
						if (src.at<uchar>(k, t))		//若此点为1，则偏移(i,j)-anchor这个位置的点设为1（取较大值的意思)
						{
							Point pos = Point(t + j, k + i) - anchor;
							if (pos.x >= 0 && pos.x < src.cols && pos.y >= 0 && pos.y < src.rows) dst.at<uchar>(pos) = 1;
						}
}

void erosion(const Mat &src, Mat &dst, const Mat &str_element)	//腐蚀
{
	Point anchor(str_element.cols / 2, str_element.rows / 2);
	dst = Mat(src.size(), src.type(), Scalar::all(1));
	for (int i = 0; i < str_element.rows; i++)
		for (int j = 0; j < str_element.cols; j++)
			if (str_element.at<uchar>(i, j))
				for (int k = 0; k < src.rows; k++)
					for (int t = 0; t < src.cols; t++)
						if (!src.at<uchar>(k, t))		//若此点为0，则偏移(i,j)-anchor这个位置的点设为0（其实是取最小值的意思)
						{
							Point pos = Point(t + j, k + i) - anchor;
							if (pos.x >= 0 && pos.x < src.cols && pos.y >= 0 && pos.y < src.rows) dst.at<uchar>(pos) = 0;
						}
}

int main()
{
	const char *filename = "sample.jpg";
	Mat src = Mat(cvLoadImage(filename, CV_LOAD_IMAGE_GRAYSCALE)) / 255;	//读入黑白图并/255（方便读入二值图片）
	Mat dst_ero, dst_dil;
	Mat str_element = getStructuringElement(cv::MORPH_ELLIPSE, Size(2 * len + 1, 2 * len + 1));	//建立structuring element
	//erode(src, dst_sample, str_element);
	erosion(src, dst_ero, str_element);
	dilation(src, dst_dil, str_element);	//尝试两个函数

	imshow("src", src * 255);
	imshow("erosion", dst_ero * 255);
	imshow("dilation", dst_dil * 255);		//显示时乘上255（方便显示）
	cvWaitKey();
	src.release();
	dst_ero.release();
	dst_dil.release();
	str_element.release();
	cvDestroyAllWindows();				//释放空间
}