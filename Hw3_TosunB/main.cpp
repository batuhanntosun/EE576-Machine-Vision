#include <iostream>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <fstream>
#include <vector> 
#include <list>
#include <complex> 
#include "hMethods.h"

using namespace std;
using namespace cv;

// Batuhan Tosun 2017401141

int main()
{
    // reading images from their directories
    string Name_img;
    Mat img = ReadImage(&Name_img);

    Mat background;
    img.copyTo(background);
    
    // convert to HSV
    cvtColor(img, img, COLOR_BGR2HSV);
    imshow("original image", background);
    waitKey(0);
    
    int total_segment;
    int reduced_total_segment = 10;
    int* ptr_seg = &total_segment;

    // get the segmented image
    Mat segmented_img;
    int threshold_method = 1;
    double threshold = 10; //18, 25
    segmented_img = ConnectedComponentAlgorithm(img, threshold_method, threshold, ptr_seg);
    cout << "Total number of labels are: " << total_segment <<endl;
    Mat color_seg;
    // reducing the segments to given number
    Mat reduced_segmented_img;
    reduced_segmented_img = ReducedSegments(img, segmented_img, total_segment, reduced_total_segment, &color_seg);
    
    // SIFT Keypoint Extractor
    SIFTextractSegment(background, reduced_segmented_img, reduced_total_segment, Name_img);

    int ellipse_threshold = 30;
    // Draw Ellipse, Find Area, Center, Second Moments
    EllipseSegment(color_seg, reduced_segmented_img, reduced_total_segment, ellipse_threshold);


    return 0;

}
