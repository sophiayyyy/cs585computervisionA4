#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"


using namespace std;
using namespace cv;

Mat trackLine(1024, 1024, CV_8UC3, Scalar(0,0,0));
RNG rng(time(0));
int objNumber = 0;

class TrackingObj
{
public:
    vector<int> currentPosition;
    vector<int> lastPosition;
    vector<vector<int>> predictP;
    vector<vector<double>> predictV;
    
    Scalar lineColor;
    
    int number = -1;
    
    double delta_T = 0.5;
    double alpha = 0.85;
    double beta = 0.005;
    
    TrackingObj(Point p){
        number = ++objNumber;
        currentPosition = {p.x, p.y};
        lastPosition = {-1, -1};
        predictP = vector<vector<int>>(2, vector<int>(2, 0));
        predictV = vector<vector<double>>(2, vector<double>(2, 0));
        lineColor = Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
    }
    
    void abFilterGetPrediction(){
        predictP[0][0] = predictP[1][0] + delta_T * predictV[1][0];
        predictP[0][1] = predictP[1][1] + delta_T * predictV[1][1];
        predictV[0] = predictV[1];
    }
    void abFilterUpdatePrediction(){
        int rx_k = currentPosition[0] - predictP[0][0];
        int ry_k = currentPosition[1] - predictP[0][1];
        predictP[0][0] += alpha * rx_k;
        predictP[0][1] += alpha * ry_k;
        predictV[0][0] += beta / delta_T * rx_k;
        predictV[0][1] += beta / delta_T * ry_k;
    }
    
    void changePrediction(){
        predictP[1] = predictP[0];
        predictV[1] = predictV[0];
    }
    
    void changePosition(int x, int y){
        lastPosition = currentPosition;
        currentPosition = {x, y};
    }
    
    void drawTrack(){
        if(lastPosition[0] == -1 && lastPosition[1] == -1)
            return;
        line(trackLine, Point(lastPosition[0], lastPosition[1]), Point(currentPosition[0], currentPosition[1]), lineColor, 2.0);
    }
};

const double MAXDISTANCE = 50.0;
const int MIN_AREA = 50;

void readText();
void readDetections();
void findCentroids();

int dataAssociate(TrackingObj obj, vector<Point>& framePoints, vector<bool>& visited);
void trackingProcess();
void mergeImg(Mat& sege);


void readImage();
void trackingProcess();
void detectionProcess();

vector<Mat> segmap; //segementation Image
vector<vector<Point>> detections;  //detections for each bat in each image
vector<Mat> rawmap; //map from Normalized folder
vector<TrackingObj> objs;



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


int main(int argc, const char * argv[]) {
    // insert code here...
    readImage();
    detectionProcess(); //deal with image to find centroids of each cell in each image
    findCentroids();
    trackingProcess();
    waitKey(0);
    return 0;
}

void readImage(){
    cv::String trainpath = "./CS585-CellImages/Normalized";
    vector<String> filenames;
    cv::glob(trainpath, filenames);
    for (size_t i=0; i< filenames.size(); i++)
    {
        if(filenames[i] == "./CellImages/Normalized/.DS_Store")
            continue;
        Mat img = imread(filenames[i], IMREAD_COLOR);
        rawmap.push_back(img);
    }
}


void trackingProcess(){
    for(int i = 0; i < detections[0].size(); i++){
        TrackingObj obj(detections[0][i]);
        obj.predictP[0] = {detections[0][i].x, detections[0][i].y};
        obj.predictP[1] = {detections[0][i].x, detections[0][i].y};
        objs.push_back(obj);
    }
    
    for(int i = 1; i < detections.size(); i++){
        vector<bool> visited(detections[i].size(), false);
        
        for(int k = 0; k < objs.size(); k++){
            objs[k].changePrediction();
            objs[k].abFilterGetPrediction();
            
            int candidateP = dataAssociate( objs[k], detections[i], visited);
            if(candidateP != -1) {
                objs[k].changePosition(detections[i][candidateP].x, detections[i][candidateP].y);
                objs[k].drawTrack();
                objs[k].abFilterUpdatePrediction();
            } else {
                // erase losting points
                //                cout << "point (" << objs[k].predictP[0][0] << ", " <<  objs[k].predictP[0][1] << ") lost" << endl;
                //                objs.erase(objs.begin()+k);
                
            }
        }
        
        for(int v = 0; v < visited.size(); v++){
            if(!visited[v]){
                TrackingObj newObj = TrackingObj(detections[i][v]);
                newObj.predictP[0] = {detections[i][v].x, detections[i][v].y};
                newObj.predictP[1] = {detections[i][v].x, detections[i][v].y};
                objs.push_back(newObj);
            }
        }
        
        Mat raw = rawmap[i];
        mergeImg(raw);
        saveImg(raw   , "tracks_" + to_string(i) + ".png");
    }
}



void mergeImg(Mat& sege){
    for(int r = 0; r < trackLine.rows; r++){
        for(int c = 0; c < trackLine.cols; c++){
            Vec3b track = trackLine.at<Vec3b>(r, c);
            if(track[0] == 0 && track[1] == 0 && track[2] == 0)
                continue;
            Vec3b se = sege.at<Vec3b>(r, c);
            se[0] = track[0]; se[1] = track[1]; se[2] = track[2];
            sege.at<Vec3b>(r, c) = se;
        }
    }
}

int dataAssociate(TrackingObj obj, vector<Point>& framePoints, vector<bool>& visited){
    cout << "obj: " <<obj.number << "\t prediction: <" << obj.predictP[0][0] << ", " << obj.predictP[0][1] << "> \t" << " currentPosition: <" << obj.currentPosition[0] << ", " << obj.currentPosition[1] << ">" <<  endl;
    int predictx = obj.predictP[0][0];
    int predicty = obj.predictP[0][1];
    
    //    bool deltax = predictx - obj.currentPosition[0] >= 0 ? true : false;
    //    bool deltay = predicty - obj.currentPosition[1] >= 0 ? true : false;
    int candidateP = -1;
    double minDistance = (double) INT_MAX;
    
    for(int i = 0; i < framePoints.size(); i++){
        if(visited[i])
            continue;
        // delta x, delta y
        //        bool cdx = framePoints[i].x - obj.currentPosition[0] > 0 ? true : false;
        //        bool cdy = framePoints[i].y - obj.currentPosition[1] > 0 ? true : false;
        double dis = sqrt(pow(double(predictx - framePoints[i].x), 2.0) + pow(double(predicty - framePoints[i].y), 2.0));
        cout << obj.number << "-" << i << ": " << dis << endl;
        //       if((cdx == deltax) && (cdy == deltay) && dis < MAXDISTANCE && (dis < minDistance)){
        if(dis < MAXDISTANCE && dis < minDistance){
            minDistance = dis;
            candidateP = i;
        }
    }
    if(candidateP == -1){
        return -1;
    }
    
    cout << obj.number << "- min" << ": " << minDistance << endl;
    visited[candidateP] = true;
    return candidateP;
}

//----------------------------------------------- (2)---------------------------------------------------//
void morphological(Mat& src) {
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    medianBlur(src, src, 5);
    erode(src, src, element);
    erode(src, src, element);
}

void morphologicalwithdilation(Mat& src) {
    Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat element2 = getStructuringElement(MORPH_RECT, Size(20, 20));
    medianBlur(src, src, 5);
    erode(src, src, element1);
    dilate(src, src, element2);
}


void findCentroids(){
    for(int i = 0; i < segmap.size(); i++){ //find all the contours for each image
        vector<vector<Point>> tempcontours;
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        Mat result;
        string centroidImg;
        findContours(segmap[i], tempcontours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0,0));
        for(int m = 0; m < tempcontours.size(); m++){
            if(tempcontours[m].size() > MIN_AREA)
                contours.push_back(tempcontours[m]);
        }
        tempcontours.clear();
        vector<Moments> moment(contours.size());
        for( int j = 0; j < contours.size(); j++){
            moment[j] = moments(contours[j], false);
        }
        vector<Point> centroids(contours.size());
        for( int k = 0; k < contours.size(); k++){
            centroids[k] = Point(moment[k].m10 / moment[k].m00, moment[k].m01 / moment[k].m00);
        }
        detections.push_back(centroids);
        Mat drawing = Mat::zeros(segmap[i].size(), CV_8UC3);
        RNG rng(time(0));
        for( int l = 0; l < contours.size(); l++){
            Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
            drawContours(drawing, contours, l, color, 2, 8, hierarchy, 0, Point());
            circle(drawing, centroids[l], 4, color, -1, 8, 0);
        }
        centroidImg = "centroid" + to_string(i+1) + ".png";
    }
}

void detectionProcess(){
    cout << "rawmap size are: " << rawmap.size() << endl;
    for(int m = 0; m < rawmap.size(); m++){
        Mat testimg = Mat::zeros(rawmap[m].size(), CV_8UC3);
        Mat testmap = Mat::zeros(rawmap[m].size(), CV_8UC3);
        cvtColor(rawmap[m], testimg, COLOR_BGR2GRAY);
        for(int i = 0; i < testimg.rows; i++){
            for(int j = 0; j < testimg.cols; j++){
                if(testimg.at<uchar>(i, j) >= 40 && testimg.at<uchar>(i, j) <= 52)
                    testimg.at<uchar>(i, j) = 0;
            }
        }
        morphological(testimg);
        for(int i = 0; i < testimg.rows; i++){
            for(int j = 0; j < testimg.cols; j++){
                if(testimg.at<uchar>(i, j) != 0){
                    testimg.at<uchar>(i, j) = 255;
                }
            }
        }
        int count = 0;
        for(int i = 0; i < testimg.rows; i++){
            for(int j = 1; j < testimg.cols; j++){
                if(testimg.at<uchar>(i, j) == 0 && testimg.at<uchar>(i, j - 1) == 255){
                    int start = j;
                    while(j < testimg.cols && testimg.at<uchar>(i, j) == 0){
                        count++;
                        j++;
                    }
                    if(count <= 20){
                        for(int k = 0; k < count; k++){
                            testimg.at<uchar>(i, start + k) = 255;
                        }
                    }
                    count = 0;
                }
            }
        }
        morphological(testimg);
        segmap.push_back(testimg);
    }
}
