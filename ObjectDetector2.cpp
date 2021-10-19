#include "ObjectDetector2.h"
#include <opencv2/imgproc.hpp>
#include <fstream> 

#include <iostream>
#include <iostream>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"


using namespace cv;
using namespace std;

ObjectDetector::ObjectDetector(const char* modelBuffer, int size, bool quantized) {
	m_modelQuantized = quantized;
	initDetectionModel(modelBuffer, size);
}

ObjectDetector::~ObjectDetector() {
	if (m_model != nullptr)
		TfLiteModelDelete(m_model);

	if (m_modelBytes != nullptr) {
		free(m_modelBytes);
		m_modelBytes = nullptr;
	}
}

void ObjectDetector::initDetectionModel(const char* modelBuffer, int size) {
	if (size < 1) { return; }

	// Copy to model bytes as the caller might release this memory while we need it
	m_modelBytes = (char*)malloc(sizeof(char) * size);
	memcpy(m_modelBytes, modelBuffer, sizeof(char) * size);

	m_model = TfLiteModelCreate(m_modelBytes, size);
	if (m_model == nullptr) {
		printf("Failed to load model");
		return;
	}

	// Build the interpreter
	TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
	TfLiteInterpreterOptionsSetNumThreads(options, 1);

	// Create the interpreter.
	m_interpreter = TfLiteInterpreterCreate(m_model, options);
	if (m_interpreter == nullptr) {
		printf("Failed to create interpreter");
		return;
	}

	// Allocate tensor buffers.
	if (TfLiteInterpreterAllocateTensors(m_interpreter) != kTfLiteOk) {
		printf("Failed to allocate tensors!");
		return;
	}

	// Find input tensors.
	if (TfLiteInterpreterGetInputTensorCount(m_interpreter) != 1) {
		printf("Detection model graph needs to have 1 and only 1 input!");
		return;
	}

	m_input_tensor = TfLiteInterpreterGetInputTensor(m_interpreter, 0);
	if (m_modelQuantized && m_input_tensor->type != kTfLiteUInt8) {
		printf("Detection model input should be kTfLiteUInt8!");
		return;
	}

	if (!m_modelQuantized && m_input_tensor->type != kTfLiteFloat32) {
		printf("Detection model input should be kTfLiteFloat32!");
		return;
	}

	if (m_input_tensor->dims->data[0] != 1 ||
		m_input_tensor->dims->data[1] != DETECTION_MODEL_SIZE ||
		m_input_tensor->dims->data[2] != DETECTION_MODEL_SIZE ||
		m_input_tensor->dims->data[3] != DETECTION_MODEL_CNLS) {
		printf("Detection model must have input dims of 1x%ix%ix%i", DETECTION_MODEL_SIZE,
			DETECTION_MODEL_SIZE, DETECTION_MODEL_CNLS);
		return;
	}
	// Find output tensors.
	if (TfLiteInterpreterGetOutputTensorCount(m_interpreter) != 1) {
		printf("Detection model graph needs to have 1 and only 1 outputs!");
		return;
	}
	m_output_classes = TfLiteInterpreterGetOutputTensor(m_interpreter, 0);
}

DetectResult* ObjectDetector::detect(Mat src) {
	DetectResult* res = new DetectResult[DETECT_NUM];
	if (m_model == nullptr) {
		return res;
	}

	Mat image;
	resize(src, image, Size(DETECTION_MODEL_SIZE, DETECTION_MODEL_SIZE), 0, 0, INTER_AREA);

	int cnls = image.type();
	if (cnls == CV_8UC1) {
		cvtColor(image, image, COLOR_GRAY2RGB);
	}
	else if (cnls == CV_8UC3) {
		cvtColor(image, image, COLOR_BGR2RGB);
	}
	else if (cnls == CV_8UC4) {
		cvtColor(image, image, COLOR_BGRA2RGB);
	}

	if (m_modelQuantized) {
		// Copy image into input tensor
		uchar* dst = m_input_tensor->data.uint8;
		memcpy(dst, image.data,
			sizeof(uchar) * DETECTION_MODEL_SIZE * DETECTION_MODEL_SIZE * DETECTION_MODEL_CNLS);
	}
	else {
		// Normalize the image so each pixel is between -1 to 1
		Mat fimage;
		image.convertTo(fimage, CV_32FC3, 2.0 / 255.0, -1.0);

		// Copy image into input tensor
		float* dst = m_input_tensor->data.f;
		memcpy(dst, fimage.data,
			sizeof(float) * DETECTION_MODEL_SIZE * DETECTION_MODEL_SIZE * DETECTION_MODEL_CNLS);
	}

	if (TfLiteInterpreterInvoke(m_interpreter) != kTfLiteOk) {
		printf("Error invoking detection model");
		return res;
	}

	const float* detection_classes = m_output_classes->data.f;
	res[0].score1 = detection_classes[0];
	res[0].score2 = detection_classes[1];

	return res;
}

Mat performDilation(Mat inputImage, int dilationElement, int dilationSize) {

	Mat outputImage;
	int dilationType;

	if (dilationElement == 0)
		dilationType = MORPH_RECT;
	
	else if (dilationElement == 1)
		dilationType = MORPH_CLOSE;
	
	else if (dilationElement == 2)
		dilationType = MORPH_ELLIPSE;

	else if (dilationElement == 3)
		dilationType = MORPH_DILATE;


	Mat element = getStructuringElement(dilationType, Size(2 * dilationSize + 1, 2 * dilationSize + 1), Point(dilationSize, dilationSize));
	dilate(inputImage, outputImage, element);

	return outputImage;

}

Mat performErosion(Mat inputImage, int erosionElement, int erosionSize) {

	Mat outputImage;
	int erosionType;

	if (erosionElement == 0)
		erosionType = MORPH_RECT;

	else if (erosionElement == 1)
		erosionType == MORPH_CLOSE;

	else if (erosionElement == 2)
		erosionType == MORPH_ERODE;

	Mat element = getStructuringElement(erosionType, Size(2 * erosionSize + 1, 2 * erosionSize + 1), Point(erosionSize, erosionSize));

	erode(inputImage,outputImage,element);
	return outputImage;

}


float calculateIoU(DetectResult bbox1, DetectResult bbox2)
{
	float iou;

	if (bbox1.xmax < bbox2.xmin || bbox2.xmax < bbox1.xmin || bbox1.ymax < bbox2.ymin || bbox2.ymax < bbox1.ymin)
	{
		iou = 0;
	}
	else {
		//Calculate the coordinates of intersection of box1 and box2.
		float x1_inter = max(bbox1.xmin, bbox2.xmin);
		float y1_inter = max(bbox1.ymin, bbox2.ymin);
		float x2_inter = min(bbox1.xmax, bbox2.xmax);
		float y2_inter = min(bbox1.ymax, bbox2.ymax);

		//Calculate intersection area.
		float inter_area = (y2_inter - y1_inter) * (x2_inter - x1_inter);

		//Calculate the Union area.
		float box1_area = (bbox1.ymax - bbox1.ymin) * (bbox1.xmax - bbox1.xmin);
		float box2_area = (bbox2.ymax - bbox2.ymin) * (bbox2.xmax - bbox2.xmin);
		float union_area = box1_area + box2_area - inter_area;

		//compute the IoU
		iou = inter_area / union_area;
	}
	return iou;
}


