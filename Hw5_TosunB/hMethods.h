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
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/ml/ml.hpp>
#include <string.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <utility>
#include <list>
#include <map>
using namespace std;
using namespace cv;
using namespace cv::ml;

// Batuhan Tosun 2017401141


// method declarations

void storeImages(string label, vector<Mat>* images);
Mat drawOptFlowMap(Mat flow, Mat cflowmap, int step);
void denseOpticalFlow(string class_name, string flow_on_image);
void featureMatching(Mat previous, Mat next, float threhshold, string window_name);
void imageStitching(string class_name, string show_each_step);
void sparseOpticalFlow(string class_name);