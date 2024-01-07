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

//a method for storing the coordinates (in total 2*N) on the images when there is a left-click action
void myCallBackFunc(int event, int x, int y, int flags, void* ptr)
{
    //vector <Point>* vec = (vector<Point>*) ptr;
    list <vector<Point>*>* d_list = (list<vector<Point>*>*) ptr;

    Point p;


    if (event == EVENT_LBUTTONDOWN) // when there is a left-click on the window
    {
        // print the corrdinates first
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;

        list <vector<Point>*>::iterator it;

        if (x > width && n2 <= N) { // if the coordinates lies on the second image
            int x_n = x - width; // actual x coordinate is found
            p = Point(x_n, y);
            it = next(d_list->begin()); // take the second element (vector) of the list
            (*it)->push_back(p); // insert that point to second vector in the list
            n2++; // increase the number of points collected from the second image

        }
        else if (x <= width && n1 <= N) { // if the coordinates lies on the first image
            p = Point(x, y);
            it = d_list->begin(); // take the first element (vector) of the list
            (*it)->push_back(p); // insert that point to first vector in the list
            n1++; // increase the number of points collected from the first image
        }

        if (n1 + n2 == 2 * N) { // when the number of points collected from both images equal to 2N
            n1 = 0;
            n2 = 0;
            destroyAllWindows(); // close the window
        }
    }
}

//a method for manuelly taking the elements of the homography map from the user and return it in matrix form
Mat homographyMapManuallyInput() {
    Mat H = Mat::zeros(3, 3, CV_64F);

    // nested for loops to get each element of the homography map separately from the user
    for (int i = 0; i < 3; i++) {
        cout << "Enter the elements in the " << i + 1 << "th row of H" << endl;
        for (int j = 0; j < 3; j++) {
            cin >> H.at<double>(i, j);
        }
    }

    return H;
}

//a method for finding and returning the homograpy map according to the given coordinates
Mat findHomographyMap(vector<Point> d, vector<Point> d_hat) {

    Mat H = findHomography(d, d_hat); // built-in opencv function to find Homography map
    cout << "Homography matrix is: " << endl;
    cout << H << endl;
    return H;

}

// a method for applying the Homography transform on a given image using to the given homography map
Mat applyHomographyTransform(Mat source, Mat desired, Mat H) {

    Mat resulting_img;
    warpPerspective(source, resulting_img, H, desired.size()); // built-in opencv function to apply Homography transform
    return resulting_img;

}

void saveCoordinates(vector<Point> d, vector<Point> d_hat) {

    // saving the coordinates to txt files
    ofstream file1("Data/img1.txt");//save coordinates to txt file;
    ofstream file2("Data/img2.txt");//save coordinates to txt file; 

    vector<Point>::iterator it = d.begin(); // to iterate over each point variable in the vector variable
    cout << "First Image Points: " << endl;
    for (; it != d.end(); ++it) {
        cout << it->x << "," << it->y << endl; // printing out
        file1 << it->x << ' ' << it->y << endl; // writing in txt
    }

    vector<Point>::iterator it2 = d_hat.begin(); // to iterate over each point variable in the vector variable
    cout << "Second Image Points: " << endl;
    for (; it2 != d_hat.end(); ++it2) {
        cout << it2->x << "," << it2->y << endl; // printing out
        file2 << it2->x << ' ' << it2->y << endl; // writing in txt
    }

}

void concatenateAndDisplay(Mat img2, Mat result1, Mat result2) {
    Mat img_all;

    // create a window to display original and transformed images
    namedWindow("My Window 2", 2);
    hconcat(img2, result1, img_all);
    hconcat(img_all, result2, img_all);
    imshow("My Window 2", img_all);
    waitKey();
}

void inputNamesImages(string* ptr1, string* ptr2) {
    // image paths
    cout << "Please enter the name of the first image: " << endl;
    cin >> *ptr1;
    cout << "Please enter the name of the second image: " << endl;
    cin >> *ptr2;
    string directory = "Data/";
    *ptr1 = directory + *ptr1;
    *ptr2 = directory + *ptr2;
    cout << *ptr1 << endl;
    cout << *ptr2 << endl;
}
