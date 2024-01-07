
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

vector<string> category_name = {"Pl1","Pl2","Pl7","Pl9","Pl10"};
vector<int> category_numbers = { 0,0,0,0,0 };

// Batuhan Tosun 2017401141
void DataReader(
    string path_txt,
    map<string, string>* PL1dataset,
    map<string, string>* PL2dataset,
    map<string, string>* PL7dataset,
    map<string, string>* PL9dataset,
    map<string, string>* PL10dataset)
{
    string img_dir;
    string img_class;
    ifstream file_(path_txt);
    category_name = { "Pl1","Pl2","Pl7","Pl9","Pl10" };
    category_numbers = { 0,0,0,0,0 };

    if (file_.is_open()) {

        while (file_ >> img_dir >> img_class) {

            if (img_class == "Pl1") {

                (*PL1dataset)[img_dir] = img_class;
                category_numbers[0] = category_numbers[0] + 1;
            }
            else if (img_class == "Pl2") {
                (*PL2dataset)[img_dir] = img_class;
                category_numbers[1] = category_numbers[1] + 1;
            }
            else if (img_class == "Pl7") {
                (*PL7dataset)[img_dir] = img_class;
                category_numbers[2] = category_numbers[2] + 1;
            }
            else if (img_class == "Pl9") {
                (*PL9dataset)[img_dir] = img_class;
                category_numbers[3] = category_numbers[3] + 1;
            }
            else if (img_class == "Pl10") {
                (*PL10dataset)[img_dir] = img_class;
                category_numbers[4] = category_numbers[4] + 1;
            }


            //cout << img_dir << "  " << img_class << endl;
        }
        file_.close();
    }

}

void vectorGenerator(int n_elements, float splitPer, vector<int>* train_numbers, vector<int>* test_numbers) {

    int n_train = int(float(n_elements) * splitPer) + 1;
    int n_test = n_elements - n_train;
    vector<int> numbers;
    for (int i = 1; i <= n_elements; i++) {
        numbers.push_back(i);
    }

    int k = 0;
    for (auto i = numbers.begin(); i != numbers.end(); ++i) {

        if (k < n_train) {
            train_numbers->push_back(*i);
        }
        else {
            test_numbers->push_back(*i);
        }
        k++;
    }
    //cout << n_train << "  " << n_test << endl;
    //cout << train_numbers->size() << "  " << test_numbers->size() << endl;
    //cout << endl;
}


void TrainTestSpliter(
    int n_categories,
    float splitPer,
    map<string, string>* train_dataset,
    map<string, string>* test_dataset,
    map<string, string>* PL1dataset,
    map<string, string>* PL2dataset,
    map<string, string>* PL7dataset,
    map<string, string>* PL9dataset,
    map<string, string>* PL10dataset)
{

    for (int i = 0; i < n_categories; i++) {

        vector<int> train_numbers;
        vector<int> test_numbers;
        vectorGenerator(category_numbers[i], splitPer, &train_numbers, &test_numbers);


        map<string, string> PiDataset;

        if (i == 0) {
            PiDataset = *PL1dataset;
        }
        else if (i == 1) {
            PiDataset = *PL2dataset;
        }
        else if (i == 2) {
            PiDataset = *PL7dataset;
        }
        else if (i == 3) {
            PiDataset = *PL9dataset;
        }
        else if (i == 4) {
            PiDataset = *PL10dataset;
        }

        int k = 1;
        for (auto pair : PiDataset) {

            if (std::find(train_numbers.begin(), train_numbers.end(), k) != train_numbers.end()) {
                (*train_dataset)[pair.first] = pair.second;
            }
            else {
                (*test_dataset)[pair.first] = pair.second;
            }
            k++;
        }
    }

    cout << "Train-test datasets" << endl;
    cout << train_dataset->size() << "  " << test_dataset->size() << endl;
    cout << endl;

}


Mat SIFTfeatureExtractor(Mat img) {

    auto detector = cv::SiftFeatureDetector::create();
    vector<KeyPoint>keypoints;
    detector->detect(img, keypoints);

    // extract the sift descriptors from image
    auto extractor = cv::SiftDescriptorExtractor::create();

    Mat descriptors;
    extractor->compute(img, keypoints, descriptors);
    //cout << descriptors.rows << " " << descriptors.cols << endl;
    return descriptors;
}

// we will read the image
// extract the sift descriptors
// and store them in a map(1) ? or add those descriptors to BOW Trainer(2) ?

void BOWRepresentation(

    map <string, string> train_dataset,
    map <string, string> test_dataset,
    map<string, Mat>* descriptor_train,
    map<string, Mat>* descriptor_test,
    map<string, Mat>* bow_train,
    map<string, Mat>* bow_test,
    int n_clusters,
    string descMatchMethod)

{

    Mat img;
    Mat SIFTdescriptors;

    BOWKMeansTrainer bowtrainer(n_clusters);

    // to obtain the words of our vocabulary
    for (auto pair : train_dataset) {
        // read the image
        img = imread(pair.first);

        // extract the SIFT features
        SIFTdescriptors = SIFTfeatureExtractor(img);
        bowtrainer.add(SIFTdescriptors);
        (*descriptor_train)[pair.first] = SIFTdescriptors;

    }

    // to obtain the words of our vocabulary
    for (auto pair : test_dataset) {
        // read the image
        img = imread(pair.first);

        // extract the SIFT features
        SIFTdescriptors = SIFTfeatureExtractor(img);
        (*descriptor_test)[pair.first] = SIFTdescriptors;

    }


    Mat vocabulary = bowtrainer.cluster();
    // not sure about the extractor :(
    Ptr<DescriptorExtractor > descExtractor = SiftDescriptorExtractor::create();
    Ptr<DescriptorMatcher > descMatcher = BFMatcher::create();
    Ptr<BOWImgDescriptorExtractor> bowExtractor = new BOWImgDescriptorExtractor(descExtractor, descMatcher);

    bowExtractor->setVocabulary(vocabulary);

    for (auto pair : *descriptor_train) {

        Mat response_hist;
        bowExtractor->compute(pair.second, response_hist);
        (*descriptor_train)[pair.first] = response_hist;
        (*bow_train)[train_dataset.at(pair.first)].push_back(response_hist);
    }

    for (auto pair : *descriptor_test) {

        Mat response_hist;
        bowExtractor->compute(pair.second, response_hist);
        (*descriptor_test)[pair.first] = response_hist;
        (*bow_test)[test_dataset.at(pair.first)].push_back(response_hist);

    }

}



void SVMTraining(int n_categories, map<string, Mat> all_samples, Ptr<SVM>* all_svms, int Upsampling) {


    for (int i = 0; i < n_categories; i++)
    {

        Mat train_samples(0, all_samples.at(category_name[i]).cols, all_samples.at(category_name[i]).type());
        Mat train_labels(0, 1, CV_32SC1);
        Mat positiveLabels(all_samples.at(category_name[i]).rows, 1, CV_32SC1, Scalar::all(1));

        if (Upsampling == 1) {
            int max_el = *max_element(category_numbers.begin(), category_numbers.end());
            int rep = int(floor(double(max_el) / double(category_numbers[i]) + 0.5));
            //cout << "For main rep: " << rep << endl;
            for (int j = 0; j < rep; j++) {
                train_samples.push_back(all_samples.at(category_name[i]));
                train_labels.push_back(positiveLabels);
            }
        }
        else {
            train_samples.push_back(all_samples.at(category_name[i]));
            train_labels.push_back(positiveLabels);
        }


        for (map<string, Mat>::iterator itr = all_samples.begin(); itr != all_samples.end(); ++itr)
        {
            if (itr->first == category_name[i]) {
                continue;
            }

            Mat negativeLabels(itr->second.rows, 1, CV_32SC1, Scalar::all(-1));

            if (Upsampling == 1) {
                int max_el = *max_element(category_numbers.begin(), category_numbers.end());
                vector<string>::iterator it = find(category_name.begin(), category_name.end(), itr->first);
                int index1 = distance(category_name.begin(), it);
                int rep = int(floor(double(max_el) / double(category_numbers[index1]) + 0.5));
                //cout << "For test rep: " << rep << endl;
                for (int j = 0; j < rep; j++) {
                    train_labels.push_back(negativeLabels);
                    train_samples.push_back(itr->second);
                }
            }

            else {
                train_labels.push_back(negativeLabels);
                train_samples.push_back(itr->second);
            }

        }

        //Setting up training parameters
        all_svms[i] = SVM::create();
        all_svms[i]->setType(SVM::C_SVC);
        all_svms[i]->setKernel(SVM::RBF);
        all_svms[i]->setGamma(0.00002);
        all_svms[i]->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
        //cout << train_samples.rows << "  " << train_samples.cols << endl;
        //cout << train_labels.rows << "   " << train_labels.cols << endl;

        all_svms[i]->train(train_samples, ROW_SAMPLE, train_labels);

    }

    cout << "Classifier training completed..." << endl;

}

string OneSVMPredict(Mat BOWvector, vector<string>* thresholded_all, int n_categories, float threshold, Ptr<SVM>* all_svms) {

    float best_score = 0.0f;
    float curConfidence;
    string prediction_category;
    string category_n;
    string thresholded_label;
    for (int i = 0; i < n_categories; i++) {

        category_n = category_name[i];



        curConfidence = all_svms[i]->predict(BOWvector, noArray(), true);
        curConfidence = 1.0 / (1.0 + exp(curConfidence));

        if (curConfidence < threshold) {
            thresholded_label = "Not";
        }
        else {
            thresholded_label = category_n;
        }

        thresholded_all->push_back(thresholded_label);

        if (curConfidence > best_score)
        {
            best_score = curConfidence;
            prediction_category = category_n;
        }


    }

    if (best_score < threshold) {
        prediction_category = "uncategorized";
    }


    return prediction_category;

}

multimap <string, vector<string>> AllSVMPredict(map<string, string>* dataset, map<string, Mat>* descriptors_dataset, int n_categories, float threshold, Ptr<SVM>* all_svms, multimap<string, string>* label_prediction) {

    string predicted_category;
    string original_category;
    multimap <string, vector<string>> label_possibilities;
    for (auto pair : *descriptors_dataset) {

        original_category = dataset->at(pair.first);
        vector<string> thresholded_all;

        predicted_category = OneSVMPredict(pair.second, &thresholded_all, n_categories, threshold, all_svms);
        label_possibilities.insert({original_category,thresholded_all});

        if (predicted_category != "uncategorized") {
            label_prediction->insert({ original_category ,predicted_category });
        }
    }
    return label_possibilities;
}

// to generate confusion matrix and get recall and precision rates for each class

void ConfMatrix(multimap<string, string> label_prediction, multimap <string, vector<string>> label_possibilities) {

    Mat ConfusionMatrix = Mat::zeros(5, 5, CV_64F);

    Mat true_positives = Mat::zeros(1, 5, CV_64F);
    Mat true_negatives = Mat::zeros(1, 5, CV_64F);
    Mat false_positives = Mat::zeros(1, 5, CV_64F);
    Mat false_negatives = Mat::zeros(1, 5, CV_64F);

    Mat precision = Mat::zeros(1, 5, CV_64F);
    Mat recall = Mat::zeros(1, 5, CV_64F);

    for (auto pair : label_prediction) {

        vector<string>::iterator it = find(category_name.begin(), category_name.end(), pair.first);
        int index1 = distance(category_name.begin(), it);
        vector<string>::iterator it2 = find(category_name.begin(), category_name.end(), pair.second);
        int index2 = distance(category_name.begin(), it2);
        ConfusionMatrix.at<double>(index1, index2) = ConfusionMatrix.at<double>(index1, index2) + 1;

    }

    Mat true_positives_models = Mat::zeros(1, 5, CV_64F);
    Mat true_negatives_models = Mat::zeros(1, 5, CV_64F);
    Mat false_positives_models = Mat::zeros(1, 5, CV_64F);
    Mat false_negatives_models = Mat::zeros(1, 5, CV_64F);
    Mat precision_models = Mat::zeros(1, 5, CV_64F);
    Mat recall_models = Mat::zeros(1, 5, CV_64F);

    for (auto pair : label_possibilities) {

        vector<string> temp_vec = pair.second;
        
        for (int i = 0; i < 5; i++) {

            // true positives
            if (pair.first == category_name[i] && temp_vec[i]==category_name[i]) {
                true_positives_models.at<double>(i) = true_positives_models.at<double>(i) + 1;
            }
            else if (pair.first == category_name[i] && temp_vec[i] != category_name[i]) {
                false_negatives_models.at<double>(i) = false_negatives_models.at<double>(i) + 1;
            }
            else if (pair.first != category_name[i] && temp_vec[i] != category_name[i]) {
                true_negatives_models.at<double>(i) = true_negatives_models.at<double>(i) + 1;
            }
            else if (pair.first != category_name[i] && temp_vec[i] == category_name[i]) {
                false_positives_models.at<double>(i) = false_positives_models.at<double>(i) + 1;
            }

        }
    }
    cout << ConfusionMatrix << endl;

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {

            if (i == j) {
                true_positives.at<double>(i) = ConfusionMatrix.at<double>(i, j);
            }
            if (i != j) {
                false_negatives.at<double>(i) = false_negatives.at<double>(i) + ConfusionMatrix.at<double>(i, j);
                false_positives.at<double>(j) = false_positives.at<double>(j) + ConfusionMatrix.at<double>(i, j);
            }
            true_negatives.at<double>(0) = true_negatives.at<double>(0) + ConfusionMatrix.at<double>(i, j);
            true_negatives.at<double>(1) = true_negatives.at<double>(1) + ConfusionMatrix.at<double>(i, j);
            true_negatives.at<double>(2) = true_negatives.at<double>(2) + ConfusionMatrix.at<double>(i, j);
            true_negatives.at<double>(3) = true_negatives.at<double>(3) + ConfusionMatrix.at<double>(i, j);
            true_negatives.at<double>(4) = true_negatives.at<double>(4) + ConfusionMatrix.at<double>(i, j);

        }
    }

    for (int i = 0; i < 5; i++) {
        true_negatives.at<double>(i) = true_negatives.at<double>(i) - true_positives.at<double>(i) - false_negatives.at<double>(i) - false_positives.at<double>(i);
    }

    for (int i = 0; i < 5; i++) {

        precision.at<double>(i) = true_positives.at<double>(i) / (true_positives.at<double>(i) + false_positives.at<double>(i) + 1e-6);
        recall.at<double>(i) = true_positives.at<double>(i) / (true_positives.at<double>(i) + false_negatives.at<double>(i) + 1e-6);

    }
    for (int i = 0; i < 5; i++) {

        precision_models.at<double>(i) = true_positives_models.at<double>(i) / (true_positives_models.at<double>(i) + false_positives_models.at<double>(i) + 1e-6);
        recall_models.at<double>(i) = true_positives_models.at<double>(i) / (true_positives_models.at<double>(i) + false_negatives_models.at<double>(i) + 1e-6);

    }

    cout << endl;
    cout << "Precision Rates: " << endl;
    cout << precision << endl;
    cout << "Recall Rates: " << endl;
    cout << recall << endl;

    cout << endl;
    cout << "Precision Model Rates: " << endl;
    cout << precision_models << endl;
    cout << "Recall Model Rates: " << endl;
    cout << recall_models << endl;

    cout << endl;


}

void imgMatcher(
    int user_number,
    map<string, string> test_dataset,
    map<string, Mat> descriptors_test,
    Ptr<SVM>* all_svms,
    map<string, string>* PL1dataset,
    map<string, string>* PL2dataset,
    map<string, string>* PL7dataset,
    map<string, string>* PL9dataset,
    map<string, string>* PL10dataset)
{
    int randomSample;
    int length_test_list = test_dataset.size();
    if (user_number == 0 or user_number>length_test_list) {
        
        randomSample = rand() % length_test_list;

    }
    else {
        randomSample = user_number;
    }
    
    string sampleLabel;
    string sampleDirectory;

    int k = 0;

    for (auto pair : test_dataset) {

        if (k == randomSample) {
            sampleDirectory = pair.first;
            sampleLabel = pair.second;
            break;
        }
        k++;
    }

    Mat sampleImg = imread(sampleDirectory);
    Mat sampleDescriptor = SIFTfeatureExtractor(sampleImg);
    Mat sampleBOWFeature = descriptors_test.at(sampleDirectory);
    string samplePrediction;
    vector<string> unnecessary;
    samplePrediction = OneSVMPredict(sampleBOWFeature, &unnecessary, 5, 0.0f, all_svms);

    map<string, string> predictedDataset;
    if (samplePrediction == "Pl1") {
        predictedDataset = *PL1dataset;
    }
    else if (samplePrediction == "Pl2") {
        predictedDataset = *PL2dataset;
    }
    else if (samplePrediction == "Pl7") {
        predictedDataset = *PL7dataset;
    }
    else if (samplePrediction == "Pl9") {
        predictedDataset = *PL9dataset;
    }
    else if (samplePrediction == "Pl10") {
        predictedDataset = *PL10dataset;
    }

    Mat matchImg, matchDescriptor;
    // matching descriptors
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_SL2);


    int biggest = 0;
    string matchDirectory;
    string matchLabel;
    for (auto pair : predictedDataset) {

        vector<vector<DMatch>> knn_matches;
        matchImg = imread(pair.first);
        matchDescriptor = SIFTfeatureExtractor(matchImg);
        matcher->knnMatch(sampleDescriptor, matchDescriptor, knn_matches, 2);

        const float ratio_thresh = 0.95f;
        vector<DMatch> good_matches;

        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        if (biggest < good_matches.size()) {

            biggest = good_matches.size();
            matchDirectory = pair.first;
            matchLabel = pair.second;
        }

    }

    cout << "The labels: " << endl;
    cout << sampleLabel << endl;
    cout << matchLabel << endl;
    cout << endl;

    cout << "The directories: " << endl;
    cout << sampleDirectory << endl;
    cout << matchDirectory << endl;
    cout << endl;

    Mat matchedImg = imread(matchDirectory);

    Mat img_all;
    namedWindow("Sample-Matched", 2);
    hconcat(sampleImg, matchedImg, img_all);
    imshow("Sample-Matched", img_all);
    waitKey();


}

