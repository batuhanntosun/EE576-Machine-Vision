#pragma once

// include section
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <fstream>
#include <vector> 
#include <list>
using namespace std;
using namespace cv;

// global variables
extern int n1;
extern int n2;
extern int N;
extern int width;

// method declarations
void myCallBackFunc(int event, int x, int y, int flags, void* ptr);
Mat homographyMapManuallyInput();
Mat findHomographyMap(vector<Point> d, vector<Point> d_hat);
Mat applyHomographyTransform(Mat source, Mat desired, Mat H);
void saveCoordinates(vector<Point> d, vector<Point> d_hat);
void concatenateAndDisplay(Mat img2, Mat result1, Mat result2);
void inputNamesImages(string* ptr1, string* ptr2);
