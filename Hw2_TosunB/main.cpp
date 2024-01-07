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
#include "hMethods.h"


using namespace cv;
using namespace std;

// Batuhan Tosun 2017401141

// global variables
int n_iteration = 2;
Scalar greenLOW = Scalar(40, 90, 10);
Scalar greenHIGH = Scalar(70, 255, 255);


int main()
{
    // reading images from their directories
    Mat img = ReadImage();
    Mat background;
    img.copyTo(background);

    // convert the image to HSV, and display the channels separately
    img = TransformAndDisplay(img);


    Mat mask, borders;
    // generate the binary mask by thresholding
    mask = GenerateApplyMask(img, background);

    // generate the borders
    borders = EnhanceAndFindBorder(background, mask);

    return 0;
}
