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

#include "hMethods.h"
using namespace std;
using namespace cv;
using namespace cv::ml;

// Batuhan Tosun 2017401141

int main()
{

    vector<float> threshold_vec = {0.0f, 0.5f, 0.8f};
    // total number of clusters <=> dimension of the BOW representation
    int n_clusters = 100;
    
    // total number of classes
    int n_categories = 5;

    // train-test split
    float splitPer = 0.75;

    // directory
    string directory = "Data/";

    // to read the txt file in which the directories and the labels of the total dataset exist
    string path_txt = "all_data.txt";

    // Match Metod
    string descMatchMethod = "FlannBased";

    // dataset stores the pairs: directory of the images as key and the corresponding class name as value
    map<string, string> train_dataset, test_dataset, PL1dataset, PL2dataset, PL7dataset, PL9dataset, PL10dataset;

    // descriptor dataset stores the pairs: directory of the images as key and the corresponding BOW feature vector as value
    map<string, Mat> descriptors_train, descriptors_test;

    // bow_all_dataset stores the pairs: (class name of the images) as key and (a matrix whose rows are the BOW feature vectors of each image within that class) as value
    map<string, Mat> bow_train, bow_test;

    
    Ptr<SVM>* all_svms = new Ptr<SVM>[n_categories];
    
    // to read the data
    cout << "Reading the data..." << endl;
    cout << endl;
    DataReader(directory+path_txt, &PL1dataset, &PL2dataset, &PL7dataset, &PL9dataset, &PL10dataset);
    cout << "Splitting the data..." << endl;
    cout << endl;
    TrainTestSpliter(n_categories, splitPer, &train_dataset, &test_dataset, &PL1dataset, &PL2dataset, &PL7dataset, &PL9dataset, &PL10dataset);
    cout << "Extracting BOW Representations..." << endl;
    cout << endl;
    BOWRepresentation(train_dataset, test_dataset, &descriptors_train, &descriptors_test, &bow_train, &bow_test, n_clusters, descMatchMethod);
    cout << "SVM Training phase..." << endl;
    cout << endl;
    SVMTraining(n_categories, bow_train, all_svms, 1);
    cout << endl;

    for (auto i = threshold_vec.begin(); i != threshold_vec.end(); ++i) {

        float threshold = *i;
        multimap<string, string> label_prediction_train;
        multimap <string, vector<string>> label_possibilities_train;

        cout << "For the threshold: " << threshold << endl;
        label_possibilities_train = AllSVMPredict(&train_dataset, &descriptors_train, n_categories, threshold, all_svms, &label_prediction_train);
        cout << "For train set: " << endl;
        ConfMatrix(label_prediction_train, label_possibilities_train);
        cout << endl;

    }

    for (auto i = threshold_vec.begin(); i != threshold_vec.end(); ++i) {

        float threshold = *i;
        multimap<string, string> label_prediction_test;
        multimap <string, vector<string>> label_possibilities_test;

        cout << "For the threshold: " << threshold << endl;
        label_possibilities_test = AllSVMPredict(&test_dataset, &descriptors_test, n_categories, threshold, all_svms, &label_prediction_test);
        cout << "For test set: " << endl;
        ConfMatrix(label_prediction_test, label_possibilities_test);
        cout << endl;

    }


    cout << endl;

    int test_sample_number = 25;
    for (int i = 1; i < 4; i++) {
        test_sample_number = test_sample_number * i;
        imgMatcher(test_sample_number, test_dataset, descriptors_test, all_svms, &PL1dataset, &PL2dataset, &PL7dataset, &PL9dataset, &PL10dataset);
    }

    
    return 0;

}
