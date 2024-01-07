
# include "hMethods.h"
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

using namespace cv;
using namespace std;

// Batuhan Tosun 2017401141

// to read the image from user and return it
Mat ReadImage() {

    string directory = "Data/";
    string name;
    cout << "Please enter the name of the first image: " << endl;
    cin >> name;

    // reading images from their directories
    Mat img = imread(directory + name);
    return img;
}

// to transform the image into HSV color space and display each channel separately
Mat TransformAndDisplay(Mat img) {

    cvtColor(img, img, COLOR_RGB2HSV);
    // separate all channels in hsv space
    vector<Mat> hsv_channels;
    split(img, hsv_channels);

    // display each channel separetly
    namedWindow("Hue Channel", 1);
    imshow("Hue Channel", hsv_channels[0]);
    namedWindow("Saturation Channel", 1);
    imshow("Saturation Channel", hsv_channels[1]);
    namedWindow("Value Channel", 1);
    imshow("Value Channel", hsv_channels[2]);
    waitKey();
    return img;

}

// to generate a binary mask using thresholding
Mat GenerateApplyMask(Mat img, Mat background) {
    Mat mask;
    inRange(img, greenLOW, greenHIGH, mask);
    mask = 255 - mask;

    Mat maskrgb, masked_img, img_all;

    background.copyTo(masked_img, mask);
    cvtColor(mask, maskrgb, COLOR_GRAY2RGB);
    // create a window to display original and transformed images
    namedWindow("Original, Mask, Masked Images", 1);
    hconcat(background, maskrgb, img_all);
    hconcat(img_all, masked_img, img_all);
    imshow("Original, Mask, Masked Images", img_all);
    waitKey();
    return mask;

}

// to enhance the resulting mask and find the borders of the main object
Mat EnhanceAndFindBorder(Mat background, Mat mask) {

    Mat enh_mask, blurred_mask, edge_mask, blurred_maskrgb, edge_maskrgb;
    // CLEAN UP RAW MASK
    int morph_size = 1;
    int morph_size2 = 1;
    Mat element = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
    Mat element2 = getStructuringElement(MORPH_RECT, Size(2 * morph_size2 + 1, 2 * morph_size2 + 1), Point(morph_size2, morph_size2));

    morphologyEx(mask, enh_mask, MORPH_OPEN, element, Point(-1, -1), n_iteration);
    morphologyEx(enh_mask, enh_mask, MORPH_CLOSE, element2, Point(-1, -1), n_iteration);

    GaussianBlur(enh_mask, blurred_mask, Size(3, 3), 3, 3);
    Canny(blurred_mask, edge_mask, 150, 255);

    cvtColor(blurred_mask, blurred_maskrgb, COLOR_GRAY2RGB);
    cvtColor(edge_mask, edge_maskrgb, COLOR_GRAY2RGB);


    Mat img_all;
    namedWindow("Original, Mask, Masked Images", 1);
    hconcat(background, blurred_maskrgb, img_all);
    hconcat(img_all, edge_maskrgb, img_all);
    imshow("Original, Enhanced Mask, Border Detected", img_all);
    waitKey();

    return edge_mask;

}
