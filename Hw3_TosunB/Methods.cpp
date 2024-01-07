
# include "hMethods.h"
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

using namespace cv;
using namespace std;

// Batuhan Tosun 2017401141

// to read the image from user and return it
Mat ReadImage(string * ptr_name) {

    string directory = "Data/";
    string name;
    cout << "Please enter the name of the first image: " << endl;
    cin >> name;
    *ptr_name = name;
    // reading images from their directories
    Mat img = imread(directory + name);
    return img;
}

// sift feature extraction part
double distance2pixels(Vec3b pixel, Vec3b npixel, int method) {

    double distance;

    // L2-Norm
    if (method == 0) {
        distance = norm(pixel - npixel);
    }

    // my distance 
    else if (method == 1) {
        double dH = 2*min(abs(pixel[0] - npixel[0]), 180 - abs(pixel[0] - npixel[0]));
        double dS = abs(pixel[1] - npixel[1]);
        double dV = abs(pixel[2] - npixel[2]);
        distance = sqrt(dH * dH + dS * dS + dV * dV);
    }

    // canberra distance
    else if (method == 2) {

        double dH = abs(pixel[0] - npixel[0]) / (pixel[0] + npixel[0]+1e-5);
        double dS = abs(pixel[1] - npixel[1]) / (pixel[1] + npixel[1]+1e-5);
        double dV = abs(pixel[2] - npixel[2]) / (pixel[2] + npixel[2]+ 1e-5);

        distance = dH + dS + dV;
    }

    // weighted euclidean distance in HSV space
    else if (method == 3) {
        double dH;
        if (pixel[0] - npixel[0] < 90) {
            dH = pixel[0] - npixel[0];
        }
        else {
            dH = 180 - pixel[0] - npixel[0];
        }
        distance = sqrt((pixel[2] - npixel[2]) * (pixel[2] - npixel[2]) + pixel[1] * pixel[1] + pixel[2] * pixel[2] - pixel[1] * npixel[1] * dH);
    }

    else if (method == 4) {
        double dH;
        if (pixel[0] - npixel[0] < 180) {
            dH = pixel[0] - npixel[0];
        }
        else {
            dH = 360 - pixel[0] - npixel[0];
        }
        double s = min(pixel[1], npixel[1]);
        double dS = abs(pixel[1] - npixel[1]) / (pixel[1] + npixel[1]+ 1e-5);
        double dV = abs(pixel[2] - npixel[2]) / (pixel[2] + npixel[2]+ 1e-5);
        distance = sqrt(s * s * dH * dH + dS * dS + dV * dV);
    }
    return distance;

}


int SIFT_featureExtractor(Mat img, string nameImg, int label) {

    auto detector = cv::SiftFeatureDetector::create();
    vector<KeyPoint>keypoints;
    detector->detect(img, keypoints);

    Mat img_w_keypoints;
    drawKeypoints(img, keypoints, img_w_keypoints);

    // extract the sift descriptors from image
    auto extractor = cv::SiftDescriptorExtractor::create();

    Mat descriptors;
    extractor->compute(img, keypoints, descriptors);

    // for visualization
    namedWindow("image with keypoints", WINDOW_AUTOSIZE);
    imshow("image with keypoints", img_w_keypoints);
    waitKey(0);
    cout << descriptors.rows << " " << descriptors.cols << endl;

    ofstream myfile;
    string file_name = "Data/" + nameImg + "Segment_" + to_string(label) + ".txt";
    myfile.open(file_name);

    int idx = 0;    
    myfile << "Total keypoints: "<< descriptors.rows << endl;
    for (KeyPoint kpt : keypoints) {

        myfile << "Keypoint Location: " << "(" << kpt.pt.x << ", " << kpt.pt.y<<")" << endl;
        myfile << descriptors.row(idx) << endl;

    }
    return descriptors.rows;
}

void drawEllipse(Mat img, Mat bin_img, int ellipse_threshold, int label, map<string,double> * ptr_map) {

    Moments mu;
    mu = moments(bin_img);

    Point2d mc = Point2d(mu.m10 / (mu.m00 + 1e-5), mu.m01 / (mu.m00 + 1e-5));
    double m11 = mu.m11 / (mu.m00 + 1e-5);
    double m20 = mu.m20 / (mu.m00 + 1e-5);
    double m02 = mu.m02 / (mu.m00 + 1e-5);

    // normalized sceond moments
    double nu20 = m20 - (mu.m10 / (mu.m00 + 1e-5)) * (mu.m10 / (mu.m00 + 1e-5));
    double nu11 = m11 - (mu.m10 / (mu.m00 + 1e-5)) * (mu.m01 / (mu.m00 + 1e-5));
    double nu02 = m02 - (mu.m01 / (mu.m00 + 1e-5)) * (mu.m01 / (mu.m00 + 1e-5));

    double delta = sqrt(4 * nu11 * nu11 - (nu20 - nu02) * (nu20 - nu02));

    double orientation = 0.5 * atan(2 * nu11 / (nu20 - nu02));

    bin_img = bin_img * 255;
    vector<vector<Point> > contours;
    findContours(bin_img, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    vector<RotatedRect> minEllipse(contours.size());

    for (size_t i = 0; i < contours.size(); i++)
    {
        if (contours[i].size() > ellipse_threshold)
        {
            minEllipse[i] = fitEllipse(contours[i]);
        }
    }

    ellipse(img, minEllipse[0], Scalar(0, 0, 255), 2);


    //ellipse(img, mc, Size(lambda2, lambda1), orientation, 0, 360, Scalar(0, 0, 255), 5);
    namedWindow("image with ellipse", WINDOW_AUTOSIZE);
    imshow("image with ellipse", img);
    waitKey(0);

    double shape_coeff = max(m20, m02) / min(m20, m02);
    cout << "Center points (x,y): " << mc << " Area: " << mu.m00 << " Second moments (m20,m11,m02): " << m20 << ", " << m11 << ", " << m02 << " Shape coeff: "<< shape_coeff<<  endl;
    
    map <string, double> ellipse_map;
   
    ellipse_map["m00"] = mu.m00;
    ellipse_map["m20"] = m20;
    ellipse_map["m11"] = m11;
    ellipse_map["m02"] = m02;
    ellipse_map["shape_coeff"] = shape_coeff;
    * ptr_map = ellipse_map;

}


void SIFTextractSegment(Mat img, Mat seg_img, int total_segment, string nameImg) {
    // for SIFT feature extraction we need to extract the segment first
    // then convert it to gray
    // then give it to sift extractor

    // generate mask from segment
    int total_features = 0;

    Mat mask;
    for (int k = 1; k <= total_segment; k++) {

        mask = (seg_img == k);
        Mat masked_img;
        img.copyTo(masked_img, mask);
        //cvtColor(masked_img, masked_img, COLOR_RGB2GRAY);
        total_features += SIFT_featureExtractor(masked_img, nameImg, k);
    }
    cout << "Total number of features: " << total_features << endl;

}
// function for ellipsing all images
void EllipseSegment(Mat img, Mat seg_img, int total_segment, int ellipse_threshold) {

    Mat masked_img;
    map <int, map<string, double>> double_map;
    
    for (int k = 1; k <= total_segment; k++) {

        Mat mask = (seg_img == k);
        map<string, double> temp;
        threshold(mask, mask, 0, 1, 0);
        drawEllipse(img, mask, ellipse_threshold, k , &temp);
        double_map[k] = temp;

    }

    double m_m00 = 0;
    double m_m20 = 0;
    double m_m11 = 0;
    double m_m02 = 0;
    double m_coeff = 0;

    for (int k = 1; k <= total_segment; k++) {
        map<string, double> temp;
        temp = double_map.at(k);
        m_m00 += temp.at("m00");
        m_m20 += temp.at("m20");
        m_m11 += temp.at("m11");
        m_m02 += temp.at("m02");
        m_coeff += temp.at("shape_coeff");

    }

    m_m00 = m_m00 / double(total_segment);
    m_m20 = m_m20 / double(total_segment);
    m_m11 = m_m11 / double(total_segment);
    m_m02 = m_m02 / double(total_segment);
    m_coeff = m_coeff / double(total_segment);

    double v_m00 = 0;
    double v_m20 = 0;
    double v_m11 = 0;
    double v_m02 = 0;
    double v_coeff = 0;

    for (int k = 1; k <= total_segment; k++) {
        map<string, double> temp;
        temp = double_map.at(k);
        v_m00 += (temp.at("m00")- m_m00) * (temp.at("m00") - m_m00);
        v_m20 += (temp.at("m20") - m_m20)* (temp.at("m20") - m_m20);
        v_m11 += (temp.at("m11") - m_m11)* (temp.at("m11") - m_m11);
        v_m02 += (temp.at("m02") - m_m02)* (temp.at("m02") - m_m02);
        v_coeff += (temp.at("shape_coeff") - m_coeff)* (temp.at("shape_coeff") - m_coeff);

    }

    cout << "Mean of Area: " << m_m00 << " Second moments (m20,m11,m02): " << m_m20 << ", " << m_m11 << ", " << m_m02 << " Shape coeff: " << m_coeff << endl;

    cout << "Variance of Area: " << v_m00/ total_segment << " Second moments (m20,m11,m02): " << v_m20/ total_segment << ", " << v_m11 / total_segment << ", " << v_m02/ total_segment << " Shape coeff: " << v_coeff/ total_segment << endl;

}



map<int, Vec3b> RandomColorMapping(int newlastLabelNo) {
    // color mapping
    map<int, Vec3b> color_map;

    for (int k = 0; k <= newlastLabelNo; k++) {

        Vec3b color = Vec3b((uchar)255 * rand() / RAND_MAX, (uchar)255 * rand() / RAND_MAX, (uchar)255 * rand() / RAND_MAX);
        color_map[k] = color;

    }
    return color_map;

}


// using those values draw an ellipse around the segment

Mat ConnectedComponentAlgorithm(Mat img, int method, double threshold, int* ptr_seg) {

    // separate all channels in hsv space
    vector<Mat> hsv_channels;
    split(img, hsv_channels);
    Mat hue_img = hsv_channels[0];

    int lastLabelNo = 0;
    Mat seg_img = Mat::zeros(hue_img.size(), CV_32SC1);

    int pairs = 0;

    // pixel value holders
    Vec3b pixel;
    Vec3b l_pixel;
    Vec3b a_pixel;

    // map to hold equivalent label values
    map<int, list<int>> map_equivalent;


    for (int r = 0; r < img.rows; r++) {

        for (int c = 0; c < img.cols; c++) {

            pixel = img.at<Vec3b>(r, c);

            // at the first row and first column
            if (r == 0 && c == 0) {
                seg_img.at<int>(r, c) = lastLabelNo;

            }

            // at the first row
            else if (r == 0 && c != 0) {

                l_pixel = img.at<Vec3b>(r, c - 1);

                if (distance2pixels(pixel, l_pixel, method) > threshold) {
                    lastLabelNo++;
                    seg_img.at<int>(r, c) = lastLabelNo;
                }
                else {
                    seg_img.at<int>(r, c) = seg_img.at<int>(r, c - 1);
                }
            }

            // at the first column
            else if (r != 0 && c == 0) {

                a_pixel = img.at<Vec3b>(r - 1, c);

                if (distance2pixels(pixel, a_pixel, method) > threshold) {
                    lastLabelNo++;
                    seg_img.at<int>(r, c) = lastLabelNo;
                }
                else {
                    seg_img.at<int>(r, c) = seg_img.at<int>(r - 1, c);
                }
            }
            // at the other regions
            else {

                l_pixel = img.at<Vec3b>(r, c - 1);
                a_pixel = img.at<Vec3b>(r - 1, c);

                if (distance2pixels(pixel, l_pixel, method) > threshold && distance2pixels(pixel, a_pixel, method) > threshold) {
                    lastLabelNo++;
                    seg_img.at<int>(r, c) = lastLabelNo;
                }

                else if (distance2pixels(pixel, l_pixel, method) > threshold && distance2pixels(pixel, a_pixel, method) <= threshold) {

                    seg_img.at<int>(r, c) = seg_img.at<int>(r - 1, c);
                }

                else if (distance2pixels(pixel, l_pixel, method) <= threshold && distance2pixels(pixel, a_pixel, method) > threshold) {

                    seg_img.at<int>(r, c) = seg_img.at<int>(r, c - 1);

                }

                else if (distance2pixels(pixel, l_pixel, method) <= threshold && distance2pixels(pixel, a_pixel, method) <= threshold) {


                    if (seg_img.at<int>(r - 1, c) == seg_img.at<int>(r, c - 1)) {

                        seg_img.at<int>(r, c) = seg_img.at<int>(r - 1, c);

                    }

                    else {

                        seg_img.at<int>(r, c) = min(seg_img.at<int>(r, c - 1), seg_img.at<int>(r - 1, c));

                        map_equivalent[min(seg_img.at<int>(r, c - 1), seg_img.at<int>(r - 1, c))].push_back(max(seg_img.at<int>(r, c - 1), seg_img.at<int>(r - 1, c)));
                        pairs++;

                    }


                }
            }
        }
    }

    cout << lastLabelNo << endl;


    // segmentation part 2

    cout << lastLabelNo << endl;
    cout << pairs << endl;


    map<int, int> map_inverse_equivalent;

    for (auto pair : map_equivalent) {

        //pair.second.unique();
        int current_label;

        // obtain the unique equivalent terms
        list<int> new_list = pair.second;
        new_list.sort();
        new_list.unique();

        if (map_inverse_equivalent.find(pair.first) == map_inverse_equivalent.end()) {
            // not found
            map_inverse_equivalent[pair.first] = pair.first;
            current_label = pair.first;
        }
        else {
            // found
            current_label = map_inverse_equivalent.at(pair.first);
        }

        list<int>::iterator it;

        for (it = new_list.begin(); it != new_list.end(); ++it) {

            if (map_inverse_equivalent.find(*it) == map_inverse_equivalent.end()) {
                // not found
                map_inverse_equivalent[*it] = current_label;
            }
        }

    }


    // segmentation part 3: full labeling

    map<int, int> map_full_equivalent;
    int newlastLabelNo = 1;

    for (int k = 0; k <= lastLabelNo; k++) {

        if (map_inverse_equivalent.find(k) == map_inverse_equivalent.end()) {
            // not found
            map_full_equivalent[k] = newlastLabelNo;
            newlastLabelNo++;

        }

        else {

            if (map_inverse_equivalent.at(k) == k) {

                map_full_equivalent[k] = newlastLabelNo;
                newlastLabelNo++;
            }
            else {

                map_full_equivalent[k] = map_full_equivalent.at(map_inverse_equivalent.at(k));

            }

        }
    }

    cout << newlastLabelNo << endl;


    // get color mapping
    map<int, Vec3b> color_map = RandomColorMapping(newlastLabelNo);
    // apply the color mapping to each segment
    Mat color_seg_img = Mat::zeros(img.size(), CV_8UC3);
    Mat last_label_img = Mat::zeros(img.size(), CV_32SC1);

    for (int r = 0; r < seg_img.rows; r++) {
        for (int c = 0; c < seg_img.cols; c++) {

            int pixel_gen = map_full_equivalent.at(seg_img.at<int>(r, c));
            last_label_img.at<int>(r, c) = pixel_gen;
            color_seg_img.at<Vec3b>(r, c) = color_map.at(pixel_gen);
            
        }
    }



    *ptr_seg = newlastLabelNo;

    imshow("Colored Segmented Image", color_seg_img);
    waitKey(0);
    imwrite("Data/col_seg_img.jpg", color_seg_img);

    return last_label_img;

}

Mat ReducedSegments(Mat img, Mat seg_img, int max_label, int max_segment, Mat * colorseg) {
    int step = 0;

    map <int, int> label2redlabel;

    map <int, list<int>> number2label;

    list<int> all_numbers;

    Mat bin_img;
    int sum_pixels;

    for (int k = 1; k < max_label + 1l; k++) {

        bin_img = (seg_img == k) / 255;
        sum_pixels = cv::sum(bin_img)[0];
        number2label[sum_pixels].push_back(k);
        all_numbers.push_back(sum_pixels);

    }

    all_numbers.sort();
    all_numbers.unique();
    all_numbers.reverse();

    list<int>::iterator it;

    for (it = all_numbers.begin(); it != all_numbers.end(); ++it) {

        int number = *it;

        list<int> label_list = number2label.at(number);
        list<int>::iterator it2;

        for (it2 = label_list.begin(); it2 != label_list.end(); ++it2) {
            label2redlabel[*it2] = step + 1;
            step++;
        }

        if (step == max_segment) {
            break;
        }

    }

    for (int k = 1; k < max_label + 1; k++) {

        if (label2redlabel.find(k) == label2redlabel.end()) {
            // not found
            label2redlabel[k] = 0;
        }
    }



    map<int, Vec3b> color_map = RandomColorMapping(max_segment);


    Mat color_seg_img = Mat::zeros(img.size(), CV_8UC3);
    Mat last_label_img = Mat::zeros(img.size(), CV_32SC1);
    for (int r = 0; r < seg_img.rows; r++) {
        for (int c = 0; c < seg_img.cols; c++) {

            int label_gen = label2redlabel.at(seg_img.at<int>(r, c));
            last_label_img.at<int>(r, c) = label_gen;
            if (label_gen == 0) {
                color_seg_img.at<Vec3b>(r, c) = Vec3b((uchar)0, (uchar)0, (uchar)0);
            }
            else {
                color_seg_img.at<Vec3b>(r, c) = color_map.at(label_gen);
            }
        }
    }

    imshow("Reduced Colored Segmented Image", color_seg_img);
    waitKey(0);
    imwrite("Data/red_col_seg_img.jpg", color_seg_img);

    *colorseg = color_seg_img;

    return last_label_img;

}

