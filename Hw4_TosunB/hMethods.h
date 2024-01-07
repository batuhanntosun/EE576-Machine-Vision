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

// global variables
extern vector<string> category_name;
extern vector<int> category_numbers;
// method declarations
void DataReader(
    string path_txt,
    map<string, string>* PL1dataset,
    map<string, string>* PL2dataset,
    map<string, string>* PL7dataset,
    map<string, string>* PL9dataset,
    map<string, string>* PL10dataset);
void vectorGenerator(int n_elements, float splitPer, vector<int>* train_numbers, vector<int>* test_numbers);
void TrainTestSpliter(
    int n_categories,
    float splitPer,
    map<string, string>* train_dataset,
    map<string, string>* test_dataset,
    map<string, string>* PL1dataset,
    map<string, string>* PL2dataset,
    map<string, string>* PL7dataset,
    map<string, string>* PL9dataset,
    map<string, string>* PL10dataset);
Mat SIFTfeatureExtractor(Mat img);
void BOWRepresentation(
    map <string, string> train_dataset,
    map <string, string> test_dataset,
    map<string, Mat>* descriptor_train,
    map<string, Mat>* descriptor_test,
    map<string, Mat>* bow_train,
    map<string, Mat>* bow_test,
    int n_clusters,
    string descMatchMethod);
void SVMTraining(int n_categories, map<string, Mat> all_samples, Ptr<SVM>* all_svms, int Upsampling);
string OneSVMPredict(Mat BOWvector, vector<string>* thresholded_all, int n_categories, float threshold, Ptr<SVM>* all_svms);
multimap <string, vector<string>> AllSVMPredict(map<string, string>* dataset, map<string, Mat>* descriptors_dataset, int n_categories, float threshold, Ptr<SVM>* all_svms, multimap<string, string>* label_prediction);
void ConfMatrix(multimap<string, string> label_prediction, multimap <string, vector<string>> label_possibilities);
void imgMatcher(
    int user_number,
    map<string, string> test_dataset,
    map<string, Mat> descriptors_test,
    Ptr<SVM>* all_svms,
    map<string, string>* PL1dataset,
    map<string, string>* PL2dataset,
    map<string, string>* PL7dataset,
    map<string, string>* PL9dataset,
    map<string, string>* PL10dataset);