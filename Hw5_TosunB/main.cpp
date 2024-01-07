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

#include "hMethods.h"
using namespace std;
using namespace cv;
using namespace cv::ml;

// Batuhan Tosun 2017401141
int main()
{

    denseOpticalFlow("human","no");
    sparseOpticalFlow("human");
    imageStitching("car","yes");
    return 0;

}
