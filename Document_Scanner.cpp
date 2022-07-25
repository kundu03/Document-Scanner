#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat imgoriginal, imgGray, imgBlur, imgCanny, imgthres, imgDilate, imgwarp, imgcrop;
vector<Point> initialpoints, docpoints;

float w = 420, h = 596;

Mat preprocessing(Mat img)
{
	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
	Canny(imgBlur, imgCanny, 25, 75);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(imgCanny, imgDilate, kernel);
	return imgDilate;
}

vector<Point> getContours(Mat image)
{
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	vector<vector<Point>> conpoly(contours.size());
	vector<Rect> boundrect(contours.size());
	vector<Point> biggest;
	int mxarea = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		int area = contourArea(contours[i]);
		if (area > 1000)
		{
			float peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], conpoly[i], 0.02 * peri, true);
			if (area > mxarea && conpoly[i].size() == 4)
			{
				biggest = { conpoly[i][0],conpoly[i][1],conpoly[i][2],conpoly[i][3] };
				mxarea = area;
			}
		}
	}
	return biggest;
}

vector<Point> reorder(vector<Point> initialpoints)
{
	vector<Point> newpoints;
	vector<int> sumpoints, subpoints;
	for (int i = 0; i < initialpoints.size(); i++)
	{
		sumpoints.push_back(initialpoints[i].x + initialpoints[i].y);
		subpoints.push_back(initialpoints[i].x - initialpoints[i].y);
	}
	newpoints.push_back(initialpoints[min_element(sumpoints.begin(), sumpoints.end()) - sumpoints.begin()]);
	newpoints.push_back(initialpoints[max_element(subpoints.begin(), subpoints.end()) - subpoints.begin()]);
	newpoints.push_back(initialpoints[min_element(subpoints.begin(), subpoints.end()) - subpoints.begin()]);
	newpoints.push_back(initialpoints[max_element(sumpoints.begin(), sumpoints.end()) - sumpoints.begin()]);
	return newpoints;
}

Mat getwarp(Mat image, vector<Point> points, float w, float h)
{
	Point2f src[4] = { points[0] , points[1] , points[2] , points[3] };
	Point2f dest[4] = { {0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h} };
	Mat matrix = getPerspectiveTransform(src, dest);
	warpPerspective(image, imgwarp, matrix, Point(w, h));
	return imgwarp;
}

int main()
{
	cout << "Welcome to SAGAR'S Project !!!!" << endl;
	cout << "This project can scan documents on a webcam." << endl;
	VideoCapture cap(0);
	while (true)
	{
		cap.read(imgoriginal);
		// Preprocessing
		imgthres = preprocessing(imgoriginal);
		// get contours
		initialpoints = getContours(imgthres); 
		docpoints = reorder(initialpoints);
		// warp
		imgwarp = getwarp(imgoriginal, docpoints, w, h);
		imshow("Image", imgoriginal);
		if (!imgwarp.empty())
		{
			imshow("doc", imgwarp);
			imwrite("Resources/doc.png", imgwarp);
			break;
		}
		waitKey(1);
	}
	return 0;
}
