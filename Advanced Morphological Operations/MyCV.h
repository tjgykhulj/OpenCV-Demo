#ifndef MYCV_H
#define MYCV_H

#include <opencv2\opencv.hpp>
using namespace cv;

const Mat default_str_element = getStructuringElement(cv::MORPH_RECT, Size(9, 9));	//建立structuring element

uchar Data[8][9] = {
	{ 0, 0, 0, 2, 1, 2, 1, 1, 1 },
	{ 0, 0, 2, 0, 1, 1, 2, 1, 1 },
	{ 1, 2, 0, 1, 1, 0, 1, 2, 0 },
	{ 2, 0, 0, 1, 1, 0, 1, 1, 2 },
	{ 1, 1, 1, 2, 1, 2, 0, 0, 0 },
	{ 1, 1, 2, 1, 1, 0, 2, 0, 0 },
	{ 0, 2, 1, 0, 1, 1, 0, 2, 1 },
	{ 2, 1, 1, 0, 1, 0, 0, 0, 2 }
};
const Mat ele[8] = { 
	Mat(3, 3, CV_8UC1, Data[0]), Mat(3, 3, CV_8UC1, Data[1]), Mat(3, 3, CV_8UC1, Data[2]), Mat(3, 3, CV_8UC1, Data[3]), 
	Mat(3, 3, CV_8UC1, Data[4]), Mat(3, 3, CV_8UC1, Data[5]), Mat(3, 3, CV_8UC1, Data[6]), Mat(3, 3, CV_8UC1, Data[7])
};

//for (int i = 0; i < 8; i++) ele[i] = Mat(3, 3, CV_8UC1, Data[i]);

int OstuThreshold(const Mat &input)
{
	double h[256] = { 0 };
	int size = input.cols * input.rows;
	double u = 0;
	for (int i = 0; i < size; i++) h[input.data[i]]++;	//计算每种深度值的点数，以及平均值sigma(i*depth[i])
	for (int i = 0; i < 256; i++) u += i*h[i] / size;
	/*
	w0 = sigma(d[i])/size;
	w1 = 1 - w0;
	u0 = sigma(i*d[i])/sigma(d[i])
	因为u = u0 * w0 + u1 * w1;
	所以u1 = (u - u0 * w0) / w1
	所以u1-u0 = (u - u0 * (w0 + w1)) / w1 = (u-u0)/w1
	得结论：g = (u1-u0)^2*w0*w1 = (u-u0)*(u-u0)/w1*w0
	*/
	double sum = 0, maxD = 0;	//sum表示sigma(i*d[i]), depth[i]表示sigma(d[i])
	int threshold;
	for (int i = 1; i < 256; i++) {
		double w0 = h[i - 1] / size;	//前景所占比例（前景设定是灰度较低的）
		double u0 = sum / h[i - 1];
		double g = (u - u0) * (u - u0) * w0 / (1 - w0);
		if (g > maxD) {
			maxD = g;
			threshold = i;	//分为0..i-1与i..255
		}
		sum += h[i] * i;
		h[i] += h[i - 1];
		if (h[i] == size) break;	//预防出现除以0的情况
	}
	return threshold;
}

int FuzzyThreshold(Mat &input)
{
	double h[256] = { 0 }, w[256];
	int size = input.rows * input.cols;
	for (int i = 0; i < size; i++) h[input.data[i]]++;

	int l = 0, r = 255;
	int threshold = -1;
	while (l < 256 && !h[l]) l++;
	while (r > l && !h[r]) r--;;

	int len = r - l + 1;
	double *s = new double[len];	//香农函数
	s[0] = 0;
	for (int i = 1; i < len; i++)
	{
		double mu = 1 / (1 + i / double(r - l));
		s[i] = -mu * log(mu) - (1 - mu) * log(1 - mu);
	}
	w[l] = l * h[l];
	for (int i = l + 1; i <= r; i++)	w[i] = w[i - 1] + i * h[i];
	double min_value = 1 << 30, sum = 0;
	for (int i = l; i < r; i++)
	{
		sum += h[i];
		double value = 0;
		int mu = round(w[i] / sum);
		for (int j = l; j <= i; j++) value += s[abs(j - mu)] * h[j];
		mu = round((w[r] - w[i]) / (size - sum));
		for (int j = i + 1; j <= r; j++) value += s[abs(j - mu)] * h[j];

		if (min_value > value)
		{
			min_value = value;      // 最小熵作为阈值
			threshold = i + 1;
		}
	}
	return threshold;
}

void Dilation(const Mat &src, const Mat &str_element)	//膨胀
{
	Point anchor(str_element.cols / 2, str_element.rows / 2);
	Mat dst(src.size(), src.type(), Scalar::all(0));
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
	for (int i = 0; i < src.cols*src.rows; i++) src.data[i] = dst.data[i];
}

void Erosion(const Mat &src, const Mat &str_element)	//腐蚀
{
	Point anchor(str_element.cols / 2, str_element.rows / 2);
	Mat dst(src.size(), src.type(), Scalar::all(1));
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
	for (int i = 0; i < src.cols*src.rows; i++) src.data[i] = dst.data[i];
}

void Opening(const Mat &src, const Mat &str_element)
{
	Erosion(src, str_element);
	Dilation(src, str_element);
}
void Closing(const Mat &src, const Mat &str_element)
{
	Dilation(src, str_element);
	Erosion(src, str_element);
}

bool HitOrMiss(const Mat &src, const Mat &str_element, int x, int y) 
{
	Point anchor(str_element.cols / 2, str_element.rows / 2);
	const uchar *data = str_element.data;
	for (int i = 0; i < str_element.rows; i++)
		for (int j = 0; j < str_element.cols; j++)
		{
			int w1 = data[i*str_element.cols + j];
			if (w1 == 2) continue;
			Point pos = Point(y + j, x + i) - anchor;
			int w2 = (pos.x >= 0 && pos.x < src.cols && pos.y >= 0 && pos.y < src.rows) ? src.data[pos.y*src.cols+pos.x] : 0;
			if (w1 == w2) continue; else return false;
		}
	return true;
}

bool Thinging(const Mat &src, int num, const Mat ele[])
{
	int rows = src.rows;
	int cols = src.cols;
	uchar *o = src.data;
	uchar *t = new uchar[rows * cols];
	bool flag = false;
	for (int k = 0; k < num; k++) {
		for (int i = 0; i < rows * cols; i++) t[i] = (o[i] && !HitOrMiss(src, ele[k], i / cols, i % cols));
		for (int i = 0; i < rows * cols; i++) if (o[i] != t[i]) flag = true, o[i] = t[i];
	}
	return flag;
}

bool Thickening(const Mat &src, int num, const Mat ele[])
{
	int size = src.cols * src.rows;
	for (int i = 0; i < size; i++) src.data[i] ^= 1;
	bool ret = Thinging(src, num, ele);
	for (int i = 0; i < size; i++) src.data[i] ^= 1;
	//把四周一圈变为0
	for (int i = 0; i < src.cols; i++) src.data[i] = src.data[size - i - 1] = src.data[size-src.cols-i-1] = 0;
	for (int i = 0; i < src.rows; i++) src.data[i*src.cols] = src.data[(i+1)*src.cols-1] = 0;
	return ret;
}

#endif