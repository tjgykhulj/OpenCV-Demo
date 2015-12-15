#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

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
			threshold = i+1;
		}
	}
	return threshold;
}

int main()
{
	int threshold;
	//读入黑白图，转为Mat，并且读出数据放在p中。
	Mat input = imread("test.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat output(input.size(), input.type());
	imshow("src", input);

	threshold = FuzzyThreshold(input);
	for (int i = 0; i < input.cols*input.rows; i++)
		output.data[i] = (input.data[i] >= threshold) * 255;
	imshow("Fuzzy", output);

	threshold = OstuThreshold(input);
	for (int i = 0; i < input.cols*input.rows; i++)
		output.data[i] = (input.data[i] >= threshold) * 255;
	imshow("Ostu", output);

	cvWaitKey(-1);
	cvDestroyAllWindows();
}