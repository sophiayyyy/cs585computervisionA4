#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include<sstream>

#include <iostream>
#include <algorithm>
#include <vector>

using namespace cv;
using namespace std;


Mat frame;
Mat fgMaskMOG2;
Mat maskContour;
Ptr<BackgroundSubtractor> pMOG2;
const static int SENSITIVITY_VALUE = 20;


void morphological(Mat& src);
Mat getAverage(vector<Mat>& inputImg);
void useBgSubtractor(vector<Mat>& inputImg);
vector<Mat> selfBgSubtrctor(vector<Mat>& inputImg, Mat& aveImg);
void findHands(vector<Mat>& inputArray, vector<Mat>& inputImg);
void getContour(Mat& src, Mat& inputImg);
void saveImg(Mat& image, string name) {
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    
    try {
        imwrite(name, image, compression_params);
    }
    catch (runtime_error& ex) {
        fprintf(stderr, "Exception converting image to JPG format: %s\n", ex.what());
    }
}

int main(int argc, char **argv) {
    vector<Mat> inputImg;
    cv::String trainpath = "./piano";
    vector<cv::String> filenames;
    cv::glob(trainpath, filenames);
    for (size_t i=0; i< filenames.size(); i++)
    {
        if(filenames[i] == "./piano/.DS_Store")
            continue;
        Mat img = imread(filenames[i], IMREAD_COLOR);
        inputImg.push_back(img);
    }
    vector<Mat> temp = inputImg;
    Mat aveImg = getAverage(inputImg);
    //    imshow("AveImg", aveImg);
    
    vector<Mat> inputXbg = selfBgSubtrctor(temp, aveImg);
    findHands(inputXbg, inputImg);
    
    waitKey(0);
    return 0;
}

//Function that returns the maximum of 3 integers
int myMax(int a, int b, int c) {
    int m = a;
    (void)((m < b) && (m = b)); //short-circuit evaluation
    (void)((m < c) && (m = c));
    return m;
}

//Function that returns the minimum of 3 integers
int myMin(int a, int b, int c) {
    int m = a;
    (void)((m > b) && (m = b));
    (void)((m > c) && (m = c));
    return m;
}
void mySkinDetect(Mat& src, Mat& dst) {
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            //For each pixel, compute the average intensity of the 3 color channels
            Vec3b intensity = src.at<Vec3b>(i, j); //Vec3b is a vector of 3 uchar (unsigned character)
            int B = intensity[0]; int G = intensity[1]; int R = intensity[2];
            if ((R > 95 && G > 40 && B > 20) && (myMax(R, G, B) - myMin(R, G, B) > 20) && (abs(R - G) > 20) && (R > G) && (R > B)) {
                dst.at<uchar>(i, j) = 255;
            }
        }
    }
}

void useBgSubtractor(vector<Mat>& inputImg){
    pMOG2 = createBackgroundSubtractorMOG2();
    for (int i = 0; i < inputImg.size(); i++) {
        frame = inputImg.at(i);
        pMOG2->apply(frame, fgMaskMOG2);
        
        stringstream ss;
        rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
                  cv::Scalar(255, 255, 255), -1);
        morphological(fgMaskMOG2);
        String outputname1 = "FG Mask MOG 2_" + to_string(i) + ".png";
        saveImg(fgMaskMOG2, outputname1);
    }
}

void morphological(Mat& src) {
    Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat element2 = getStructuringElement(MORPH_RECT, Size(20, 20));
    medianBlur(src, src, 5);
    erode(src, src, element1);
    dilate(src, src, element2);
}

Mat getAverage(vector<Mat>& inputImg){
    Mat aveImg  = inputImg[0];
    int size = (int)inputImg.size();
    int rows = aveImg.rows;
    int cols = aveImg.cols;
    
    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            int blue = 0, green = 0, red = 0;
            for(int i = 0; i < inputImg.size(); i++){
                c = (c >= inputImg[i].cols) ? inputImg[i].cols-1 : c;
                r = (r >= inputImg[i].rows) ? inputImg[i].rows-1 : r;
                Vec3b input = inputImg[i].at<Vec3b>(r,c);
                blue += input[0];
                green += input[1];
                red += input[2];
            }
            Vec3b ave = aveImg.at<Vec3b>(Point(c,r));
            ave[0] = blue / size;
            ave[1] = green / size;
            ave[2] = red / size;
            aveImg.at<Vec3b>(Point(c,r)) = ave;
        }
    }
    return aveImg;
}
vector<Mat> selfBgSubtrctor(vector<Mat>& inputImg, Mat& aveImg){
    vector<Mat> res;
    for(int i = 0; i < inputImg.size(); i++){
        Mat imgXbg = Mat::zeros(inputImg[i].rows, inputImg[i].cols, CV_8UC1);
        absdiff(inputImg[i], aveImg, imgXbg);
        Mat saveSkin;
        saveSkin = Mat::zeros(inputImg[i].rows, inputImg[i].cols, CV_8UC1);
        mySkinDetect(imgXbg, saveSkin);
        morphological(saveSkin);
        //        saveImg(saveSkin, "Xbg_skin_"+to_string(i)+".png");
        res.push_back(saveSkin);
    }
    return res;
}

bool comparePoint(Point p1, Point p2){
    return p1.x < p2.x;
}

bool compareLeftMost(pair<Point, int> p1, pair<Point, int> p2){
    return p1.first.x < p2.first.x;
}

bool compareArea(pair<double, int> contour1, pair<double, int> contour2){
    return contour1.first > contour2.first;
}

void getContour(Mat& src, Mat& inputImg) {
    // find all contours
    Mat thres_output;
    threshold(src, thres_output, SENSITIVITY_VALUE, 255, THRESH_BINARY);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(thres_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    
    vector<pair<Point, int>> leftMost;
    vector<int> contourInd;
    vector<pair<double, vector<Point>>> areaContours;
    
    // find the left most point of all points which a contour has
    for (int i = 0; i < contours.size(); i++)
    {
        vector<Point> curContour = contours[i];
        vector<pair<int, int>> points;
        std::sort(curContour.begin(), curContour.end(), comparePoint);
        leftMost.push_back(make_pair(curContour.front(), i));
    }
    
    // sort all contours based on their left most point's x
    std::sort(leftMost.begin(), leftMost.end(), compareLeftMost);
    
    for(int i = 0; i < leftMost.size(); i++){
        Rect bound = boundingRect(contours[leftMost[i].second]);
        rectangle(inputImg, bound, Scalar(0, 0, 255), 2, 8, 0);
        putText(inputImg, "left_"+to_string(i), Point(bound.x, bound.y), FONT_HERSHEY_PLAIN, 3.0, CV_RGB(255,0,0));
    }
    
    // find the distances between the point and its previous one to get the start point and end point
    vector<pair<int, int>> leftDiff;
    int start = 0, end = 0;
    for(int i = 1; i < leftMost.size(); i++){
        leftDiff.push_back({leftMost[i].first.x - leftMost[i-1].first.x, i});
    }
    cout << "left diff : " << endl;
    for(int i = 0; i < leftDiff.size(); i++){
        cout << to_string(leftDiff[i].second-1) + "->" + to_string(leftDiff[i].second)+ ": "<< leftDiff[i].first << endl;
        if(leftDiff[i].first > 60){
            if(i > 0){
                end = leftDiff[i].second-1;
                break;
            }
            else
                start = leftDiff[i].second;
        }
    }
    cout << "start = " << start << ", end = " << end << endl;
    
    vector<pair<double, int>> leftArea;
    for(int i = start; i <= end; i++){
        vector<Point> curArea = contours[leftMost[i].second];
        leftArea.push_back({contourArea(curArea), leftMost[i].second});
    }
    std::sort(leftArea.begin(), leftArea.end(), compareArea);
    
    if(leftArea.size() > 1){
        int y1 = leftMost[leftArea[0].second].first.x;
        int y2 = leftMost[leftArea[1].second].first.x;
        
        Rect bound1 = boundingRect(contours[leftArea[0].second]);
        Rect bound2 = boundingRect(contours[leftArea[1].second]);
        
        auto left = Scalar(0, 255, 0), right = Scalar(255, 0, 0);
        
        if(y1 > y2){
            rectangle(inputImg, bound1, right, 2, 8, 0);
            putText(inputImg, "right", Point(bound1.x, bound1.y), FONT_HERSHEY_PLAIN, 3.0, right);
            rectangle(inputImg, bound2, left, 2, 8, 0);
            putText(inputImg, "left", Point(bound2.x, bound2.y), FONT_HERSHEY_PLAIN, 3.0, left);
        } else {
            rectangle(inputImg, bound2, right, 2, 8, 0);
            putText(inputImg, "right", Point(bound2.x, bound2.y), FONT_HERSHEY_PLAIN, 3.0, right);
            rectangle(inputImg, bound1, left, 2, 8, 0);
            putText(inputImg, "left", Point(bound1.x, bound1.y), FONT_HERSHEY_PLAIN, 3.0, left);
        }
    }
}

void findHands(vector<Mat>& inputArray, vector<Mat>& inputImg){
    for(int i = 1; i < inputArray.size(); i++){
        cout << "-------------------"<<" img" + to_string(i) << "-------------------" << endl;
        getContour(inputArray[i], inputImg[i]);
        saveImg( inputImg[i], "findHands_"+to_string(i)+".png");
    }
}

