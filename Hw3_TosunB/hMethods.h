#pragma once

// include section
#include <iostream>
#include <map>
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


// method declarations
Mat ReadImage(string* ptr_name);
double distance2pixels(Vec3b pixel, Vec3b npixel, int method);
int SIFT_featureExtractor(Mat img, string nameImg, int label);
void drawEllipse(Mat img, Mat bin_img, int ellipse_threshold, int label, map<string, double>* ptr_map);
map<int, Vec3b> RandomColorMapping(int newlastLabelNo);
Mat ConnectedComponentAlgorithm(Mat img, int method, double threshold, int* ptr_seg);
void SIFTextractSegment(Mat img, Mat seg_img, int total_segment, string nameImg);
void EllipseSegment(Mat img, Mat seg_img, int total_segment, int ellipse_threshold);
Mat ReducedSegments(Mat img, Mat seg_img, int max_label, int max_segment, Mat* colorseg);