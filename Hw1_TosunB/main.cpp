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

// global variables
int n1 = 0; // the number of coordinates chosen in img1
int n2 = 0; // the number of coordinates chosen in img2
int N=0; // the number N of input corresponding points:
int width=0; // width of the img1

int main()
{
    string path1;
    string path2;;

    // take the names of the inputs
    inputNamesImages(&path1, &path2);

    // reading images from their directories
    Mat img1 = imread(path1);
    Mat img2 = imread(path2);

    width = img1.cols;
    cout << "The width of the image is: " << width << endl;

    // taking the number of input corresponding points
    int N_user;
    cout << "Enter the number N of input corresponding points: " << endl;
    cin >> N_user;
    N = N_user; // assigning to global variable

    // to create joint images side-by-side
    Mat joint_images;
    hconcat(img1, img2, joint_images);
    //check if fail to read the image
    if (img1.empty() or img2.empty())
    {
        cout << "Error loading the images" << endl;
        return -1;
    }

    // defining d and d_hat vectors to hold chosen coordinates in img1 and img2
    vector <Point> d;
    vector <Point> d_hat;

    // defining pointers for holding the address of two vectors
    vector <Point>* d_ptr = &d;
    vector <Point>* d_hat_ptr = &d_hat;

    // creating a list for holding the vector pointers
    list<vector<Point>*> d_list;
    d_list.push_back(d_ptr);
    d_list.push_back(d_hat_ptr);

    // create a window
    namedWindow("My Window", 1);
    // show the joint images
    imshow("My Window", joint_images);
    // set the callback function for any mouse event
    setMouseCallback("My Window", myCallBackFunc, &d_list);
    waitKey(0);

    // saving the coordinates to txt files
    saveCoordinates(d, d_hat);

    Mat H_cv = findHomographyMap(d, d_hat); // homogprahy map obtained with built-in opencv function
    Mat H = homographyMapManuallyInput(); // homogprahy map manually obtained

    Mat result1 = applyHomographyTransform(img1, img2, H_cv); 
    Mat result2 = applyHomographyTransform(img1, img2, H);

    // create a window to display original and transformed images
    concatenateAndDisplay(img2, result1, result2);

    return 0;
}
