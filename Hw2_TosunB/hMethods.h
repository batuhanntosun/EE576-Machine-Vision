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

// Batuhan Tosun 2017401141

// global variables

extern int n_iteration;
extern Scalar greenLOW;
extern Scalar greenHIGH;

// method declarations

Mat ReadImage();
Mat TransformAndDisplay(Mat img);
Mat GenerateApplyMask(Mat img, Mat background);
Mat EnhanceAndFindBorder(Mat background, Mat mask);
