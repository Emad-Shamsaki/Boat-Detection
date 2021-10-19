#pragma once
#include <opencv2/core.hpp>
#include "tensorflow/lite/c/c_api.h"
#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>



using namespace cv;


struct DetectResult {
	float score1 = 0;
	float score2 = 0;
	float ymin = 0.0;
	float xmin = 0.0;
	float ymax = 0.0;
	float xmax = 0.0;
};



class ObjectDetector {
public:
	ObjectDetector(const char* modelBuffer, int size, bool quantized = false);
	~ObjectDetector();
	DetectResult* detect(Mat src);
	const int DETECT_NUM = 1;
private:
	// members
	const int DETECTION_MODEL_SIZE = 224;
	const int DETECTION_MODEL_CNLS = 3;
	bool m_modelQuantized = false;
	char* m_modelBytes = nullptr;
	TfLiteModel* m_model;
	TfLiteInterpreter* m_interpreter;
	TfLiteTensor* m_input_tensor = nullptr;
	const TfLiteTensor* m_output_classes = nullptr;

	// Methods
	void initDetectionModel(const char* modelBuffer, int size);
};

float calculateIoU(DetectResult bbox1, DetectResult bbox2);
Mat performDilation(Mat inputImage, int dilationElement, int dilationSize);
Mat performErosion(Mat inputImage, int erosionElement, int erosionSize);
