
# include "hMethods.h"
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

using namespace cv;
using namespace std;
using namespace cv::ml;



void storeImages(string label, vector<Mat>* images) {

    vector<cv::String> fn;
    if (label == "car") {
        cv::glob("Data/Car2/img/*.jpg", fn, false);
        size_t count = fn.size(); //number of png files in images folder
        for (size_t i = 0; i < count; i++) {
            images->push_back(imread(fn[i]));
        }
    }
    else if (label == "human") {
        cv::glob("Data/Human9/img/*.jpg", fn, false);
        size_t count = fn.size(); //number of png files in images folder
        for (size_t i = 0; i < count; i++) {
            images->push_back(imread(fn[i]));
        }
    }
    else {
        cout << "ERROR: The category could not be found!" << endl;
    }

}



Mat drawOptFlowMap(Mat flow, Mat temp, int step) {

    for (int y = 0; y < temp.rows; y += step) {
        for (int x = 0; x < temp.cols; x += step)
        {
            Point2f f_xy = flow.at<Point2f>(y, x);
            line(temp, Point(x, y), Point(cvRound(x + f_xy.x), cvRound(y + f_xy.y)), Scalar(255, 255, 0));
            circle(temp, Point(x, y), 1.2, Scalar(255, 255, 0), -1);

        }
    }
    return temp;
}


void denseOpticalFlow(string class_name, string flow_on_image) {

    vector<Mat> images;
    storeImages(class_name, &images);
    int size_images = images.size();
    cout << "The number of car images is : " << size_images << endl;
    int step = 10;
    Mat frame0, previous;
    frame0 = images[0];
    // convert to gray
    cvtColor(frame0, previous, COLOR_BGR2GRAY);

    namedWindow("Dense Optical Flow", 1);

    for (int i = 1; i < size_images; i++) {
        Mat frame1, next;
        frame1 = images[i];
        cvtColor(frame1, next, COLOR_BGR2GRAY);

        Mat flow(previous.size(), CV_32FC2);

        calcOpticalFlowFarneback(previous, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        Mat temp;

        Mat temp_empty = Mat::zeros(flow.size(), CV_8UC3);
        if (flow_on_image == "yes") {
            temp = drawOptFlowMap(flow, frame0, step);
        }

        else {
            temp_empty = drawOptFlowMap(flow, temp_empty, step);
        }

        // visualization
        Mat flow_parts[2];
        split(flow, flow_parts);
        Mat magnitude, angle, magn_norm;
        cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
        normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
        angle *= ((1.f / 360.f) * (180.f / 255.f));

        // build hsv image
        Mat hsv_img[3], hsv_all, hsv_eight, bgr_img;
        hsv_img[0] = angle;
        hsv_img[1] = Mat::ones(angle.size(), CV_32F);
        hsv_img[2] = magn_norm;
        merge(hsv_img, 3, hsv_all);
        hsv_all.convertTo(hsv_eight, CV_8U, 255.0);
        cvtColor(hsv_eight, bgr_img, COLOR_HSV2BGR);

        Mat img_all, img_temp;
        if (flow_on_image == "yes") {
            hconcat(temp, bgr_img, img_all);
        }
        else {
            hconcat(frame0, temp_empty, img_temp);
            hconcat(img_temp, bgr_img, img_all);
        }


        cv::imshow("Dense Optical Flow", img_all);

        // also draw the vectors :(

        cv::waitKey();

        previous = next;
        frame0 = frame1;

    }

}



void featureMatching(Mat previous, Mat next, float threhshold, string window_name) {

    // SIFT part
    auto detector = cv::SiftFeatureDetector::create();
    auto extractor = cv::SiftDescriptorExtractor::create();

    Mat descriptors_prev, descriptors_next;

    vector<KeyPoint> kp_prev, kp_next;
    detector->detect(previous, kp_prev);
    extractor->compute(previous, kp_prev, descriptors_prev);

    detector->detect(next, kp_next);
    extractor->compute(next, kp_next, descriptors_next);


    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_SL2);

    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch(descriptors_prev, descriptors_next, knn_matches, 2);

    // Filter matches using the Lowe's ratio test
    float ratio_thresh = threhshold;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    Mat img_matches;
    drawMatches(previous, kp_prev, next, kp_next, good_matches, img_matches, Scalar::all(-1),
        Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //-- Show detected matches
    imshow(window_name, img_matches);

}



void imageStitching(string class_name, string show_each_step) {

    vector<Mat> images;
    storeImages(class_name, &images);
    int size_images = images.size();
    cout << "The number of car images is : " << size_images << endl;

    Mat frame0, previous;
    frame0 = images[0];
    // convert to gray
    cvtColor(frame0, previous, COLOR_BGR2GRAY);

    // SIFT part
    auto detector = cv::SiftFeatureDetector::create();
    auto extractor = cv::SiftDescriptorExtractor::create();

    vector<KeyPoint> kp_prev, kp_next;
    vector<Point2f> p_prev, p_next;


    Mat descriptors_prev, descriptors_next;
    // extract the SIFT keypoints and descriptors of the previus frame
    detector->detect(previous, kp_prev);
    extractor->compute(previous, kp_prev, descriptors_prev);
    KeyPoint::convert(kp_prev, p_prev);

    namedWindow("Image Stitching", 1);
    namedWindow("Good Matches", 1);

    Mat previous_m, next_m;
    previous_m = previous;
    Mat warped_img;

    for (int i = 1; i < size_images; i++) {

        Mat frame1, next;
        frame1 = images[i];

        cvtColor(frame1, next, COLOR_BGR2GRAY);
        next_m = next;

        detector->detect(next, kp_next);
        extractor->compute(next, kp_next, descriptors_next);


        KeyPoint::convert(kp_next, p_next);

        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_SL2);

        std::vector< std::vector<DMatch> > knn_matches;
        matcher->knnMatch(descriptors_prev, descriptors_next, knn_matches, 2);

        // Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.9f;
        std::vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }



        vector<Point2f> imgPointsPrev;
        vector<Point2f> imgPointsNext;

        cout << "Done until here " << endl;

        if (good_matches.size() > 4) {

            for (size_t j = 0; j < good_matches.size(); j++) {


                DMatch temp1 = good_matches[j];
                imgPointsPrev.push_back(p_prev[temp1.queryIdx]);
                imgPointsNext.push_back(p_next[temp1.trainIdx]);

            }
        }

        cout << "Done until here " << endl;
        Mat H = findHomography(imgPointsPrev, imgPointsNext, RANSAC);


        warpPerspective(frame0, warped_img, H, Size(previous.cols + next.cols, previous.rows + next.rows));


        cv::Mat half(warped_img, cv::Rect(0, 0, next.cols, next.rows));
        frame1.copyTo(half);

        Mat warped_img_gray;
        cvtColor(warped_img, warped_img_gray, COLOR_BGR2GRAY);

        // //Finding the largest contour i.e remove the black region from image

        threshold(warped_img_gray, warped_img_gray, 0, 255, THRESH_BINARY);
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(warped_img_gray, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
        int largest_area = 0;
        int largest_contour_index = 0;
        Rect bounding_rect;

        for (int i = 0; i < contours.size(); i++) 
        {
            double a = contourArea(contours[i], false);  
            if (a > largest_area) {
                largest_area = a;
                largest_contour_index = i; 
                bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
            }

        }

        // Scalar color( 255,255,255);
        warped_img = warped_img(Rect(bounding_rect.x, bounding_rect.y, bounding_rect.width, bounding_rect.height));

        if (show_each_step == "yes") {
            featureMatching(previous_m, next_m, ratio_thresh, "Good Matches");
            cv::imshow("Image Stitching", warped_img);
            cv::waitKey();
        }

        frame0 = warped_img;
        cvtColor(warped_img, previous, COLOR_BGR2GRAY);
        previous_m = next_m;
        detector->detect(previous, kp_prev);
        extractor->compute(previous, kp_prev, descriptors_prev);
        KeyPoint::convert(kp_prev, p_prev);

    }

    cv::imwrite("Data/stitched_img_" + class_name + ".jpg", warped_img);
    cv::imshow("Final", warped_img);
    cv::waitKey();

}


void sparseOpticalFlow(string class_name) {

    // Create some random colors
    vector<Scalar> colors;
    RNG rng;
    for (int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r, g, b));
    }


    /*
    // SIFT part
    auto detector = cv::SiftFeatureDetector::create();
    auto extractor = cv::SiftDescriptorExtractor::create();
    vector<KeyPoint> kp_prev, kp_next;
    vector<Point2f> p_prev, p_next;
    Mat descriptors_prev, descriptors_next;
    */

    vector<Mat> images;
    storeImages(class_name, &images);
    int size_images = images.size();
    cout << "The number of car images is : " << size_images << endl;

    Mat frame0, previous;
    frame0 = images[0];
    // convert to gray
    cvtColor(frame0, previous, COLOR_BGR2GRAY);

    /*
    // extract the SIFT keypoints and descriptors of the previus frame
    detector->detect(previous, kp_prev);
    extractor->compute(previous, kp_prev, descriptors_prev);
    KeyPoint::convert(kp_prev, p_prev);
    */

    vector <Point2f> p_prev, p_next;
    goodFeaturesToTrack(previous, p_prev, 100, 0.3, 7, Mat(), 7, false, 0.04);

    namedWindow("Good Matches", 1);

    // Create a mask image for drawing purposes
    Mat mask = Mat::zeros(frame0.size(), frame0.type());



    for (int i = 1; i < size_images; i++) {


        Mat frame1, next;
        frame1 = images[i];
        cvtColor(frame1, next, COLOR_BGR2GRAY);

        // calculate optical flow
        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(previous, next, p_prev, p_next, status, err, Size(15, 15), 2, criteria);
        vector<Point2f> good_new;


        for (uint i = 0; i < p_prev.size(); i++)
        {
            // Select good points
            if (status[i] == 1) {
                good_new.push_back(p_next[i]);
                // draw the tracks
                line(mask, p_next[i], p_prev[i], colors[i], 2);
                circle(frame1, p_next[i], 5, colors[i], -1);
            }
        }

        Mat img;
        cv::add(frame1, mask, img);
        cv::imshow("Good Matches", img);
        cv::waitKey();

        previous = next.clone();
        p_prev = good_new;
    }

}
