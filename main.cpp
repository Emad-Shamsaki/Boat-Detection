#include<opencv2/core/core.hpp>
#include <fstream>
#include "ObjectDetector2.h"
#include"selective_search.hpp"
#include "nms.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	//Variables
		//path for image 
	string path;
		//IoU validation
	array<DetectResult, 100>bbox1;
	array<DetectResult, 100>bbox2;
	char ans;
	int n, m;
	m = 0;
	array<float, 100> iou;
	float iou_max;
	float xmin, ymin, xmax, ymax;
	vector<vector<float>> proposals, final_proposals;

	  //load trained network
	ifstream ifile("../Project5/tfLite_MODEL .tflite", ifstream::binary);
	ifile.seekg(0, ifile.end);
	int length = ifile.tellg();
	ifile.seekg(0, ifile.beg);
	char* modelBuffer = new char[length];
	ifile.read(modelBuffer, length);
	ObjectDetector detector = ObjectDetector(modelBuffer, length, false);
	DetectResult* res;

	//imread
	cout << "Welcome to Boat detection Software" << endl;
	cout << "Please Enter your Image path:"<<endl;
	getline(cin, path);
	//path = "C:\\Users\\emads\\Documents\\2\\com vision\\final project\\DATA fo teat\\Kaggle_ships\\Kaggle_ships\\01.jpg";
	cout << "You entered: " << path << endl << endl;

	Mat img = imread(path);
	if (img.empty()) {
		cout << "no image in this path";
	}

	//Filtering and Sharpening
	Mat gaussBlur;
	//bilateralFilter(img, gaussBlur, 15, 80, 80, BORDER_DEFAULT);
	GaussianBlur(img, gaussBlur, Size(5, 5), 0);
	addWeighted(img, 8.5, gaussBlur, -7.5, 0, gaussBlur);

	//Selective Search
	auto ssproposals = ss::selectiveSearch(gaussBlur, 500, 0.8, 50, 1000, 50000, 10);
	for (auto&& rect : ssproposals) {
		xmin = rect.x;
		xmax = xmin + rect.width;
		ymin = rect.y;
		ymax = rect.y + rect.height;
		proposals.push_back({ xmin,ymin,xmax,ymax });
	}

	//Non max suppression
	std::vector<cv::Rect> prop_boxes = nms(proposals, 0.5);
	for (int i = 0; i < prop_boxes.size(); i++) {
		rectangle(gaussBlur, prop_boxes[i], cv::Scalar(0, 255, 0), 3, 8);
	}


	//Feed the regions of interest to the trained network
	if (prop_boxes.size() != 0) {
		for (int i = 0; i < prop_boxes.size(); i++) {
			res = detector.detect(img(prop_boxes[i]));
			float score1 = res[0].score1;
			float score2 = res[0].score2;
			if (score1 > 0.9) { //You can modify the threshold to 0.75 
				xmin = prop_boxes[i].x;
				xmax = xmin + prop_boxes[i].width;
				ymin = prop_boxes[i].y;
				ymax = prop_boxes[i].y + prop_boxes[i].height;
				final_proposals.push_back({ xmin,ymin,xmax,ymax });
			}
		}
	}

	//Non max suppression to get final results
	std::vector<cv::Rect> final_prop_boxes = nms(final_proposals, 0.1);
	
	//Draw the bounding boxes
	for (int i = 0; i < final_prop_boxes.size(); i++) {
		rectangle(img, final_prop_boxes[i], Scalar(0, 255, 0), 2);
		bbox2[m].xmin = final_prop_boxes[i].x;
		bbox2[m].ymin = final_prop_boxes[i].y;
		bbox2[m].xmax = bbox2[m].xmin + final_prop_boxes[i].width;
		bbox2[m].ymax = bbox2[m].ymin + final_prop_boxes[i].height;
		m++;
	}



	//Do you want an IoU evaluation?
	cout << "Do you want to evaluate the detection by using IoU?(yes/no)" << endl;
	cin >> ans;
	if (ans == 'y' || ans == 'Y' || ans == 'yes' || ans == 'YES' || ans == 'Yes') {
		cout << "Enter the number of Objects that you have the lables" << endl;
		cin >> n;
		for (int i = 0; i < n; ++i) {
			cout << "enter object number " << i + 1 << " coordinates:" << endl;
			cout << "x min" << endl;
			cin >> bbox1[i].xmin;
			cout << "y min" << endl;
			cin >> bbox1[i].ymin;
			cout << "x max" << endl;
			cin >> bbox1[i].xmax;
			cout << "y max" << endl;
			cin >> bbox1[i].ymax;
		}


		for (int i = 0; i < n; ++i) {
			iou_max = 0;
			for (int j = 0; j < m; ++j) {
				iou[j] = calculateIoU(bbox1[i], bbox2[j]);
				if (iou[j] > iou_max) {
					iou_max = iou[j];
				}
			}

			rectangle(img, Point(bbox1[i].xmin, bbox1[i].ymin), Point(bbox1[i].xmax, bbox1[i].ymax), Scalar(255, 0, 0), 2);
			if (iou_max != 0) {
				putText(img, "boat_IoU: " + to_string(iou_max), Point(bbox1[i].xmin, bbox1[i].ymin), FONT_HERSHEY_PLAIN, 1.5, Scalar(255, 0, 0), 2);
			}
			else {
				putText(img, "no boat detected ", Point(bbox1[i].xmin, bbox1[i].ymin), FONT_HERSHEY_PLAIN, 1, Scalar(255, 0, 0), 2);

			}
		}
	}

	imshow("result", img);
	waitKey(0);
	return 0;
}