#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <boost/shared_ptr.hpp>
#include <opencv2/cudaobjdetect.hpp>

#include<device_launch_parameters.h>
#include<device_functions.h>
#include <cuda_runtime.h>

#include <vector>
#include <math.h>
#include <iostream>
#include <string>

#include<caffe/layers/input_layer.hpp>
#include<caffe/layers/concat_layer.hpp>
#include<caffe/layers/inner_product_layer.hpp>
#include<caffe/layers/dropout_layer.hpp>
#include<caffe/layers/flatten_layer.hpp>
#include<caffe/layers/sigmoid_layer.hpp>
#include<caffe/layers/reduction_layer.hpp>
#include<caffe/layers/batch_norm_layer.hpp>
#include<caffe/layers/scale_layer.hpp>
#include<caffe/layers/bias_layer.hpp>
#include<caffe/layers/prelu_layer.hpp>
#include<caffe/layers/axpy_layer.hpp>

#include <time.h>
namespace caffe {
	extern INSTANTIATE_CLASS(InputLayer);
	extern INSTANTIATE_CLASS(ConcatLayer);
	extern INSTANTIATE_CLASS(InnerProductLayer);
	extern INSTANTIATE_CLASS(DropoutLayer);
	extern INSTANTIATE_CLASS(FlattenLayer);
	extern INSTANTIATE_CLASS(SigmoidLayer);
	extern INSTANTIATE_CLASS(ReductionLayer);
	extern INSTANTIATE_CLASS(BatchNormLayer);
	extern INSTANTIATE_CLASS(ScaleLayer);
	extern INSTANTIATE_CLASS(BiasLayer);
	extern INSTANTIATE_CLASS(PReLULayer);
	extern INSTANTIATE_CLASS(AxpyLayer);
}

#define  PI 3.1415926


using caffe::Blob;
using caffe::Net;
using namespace cv;
using namespace std;


extern std::vector<float> getAllangle(std::vector<const float*> set,const float* _f);
/**
@brief 基于Arcface的人脸去重
*/
class Arcface{
public:
	/**
	@brief 定义同一个人的脸
	*/
	struct Face {
		cv::Point2f landmark[5];
		/**
		@brief 人脸集合中最好的图片的置信度，即人脸评分
		*/
		float confidence;
		const float* markfeature;
		static int count;
		int faceid;
		/**
		@brief 该人脸的所有特征
		*/
		vector<float*> feature;
		/**
		@brief 人脸集合中最好的图片
		*/
		vector<Mat> faceimg;
		Face() {
			faceid = count;
			count++;
			if (count == INT16_MAX) count = 0;
		}
	};
	//初始化网络
	void init();
	//测试函数,可忽略
	float getAngle(Mat _m,Mat _n);
	/**@brief 输入图片，后得到该图片的ID
	@param 所输入的人脸图片
	@return 返回人脸ID
	*/
	int getId(Mat m);
	//测试函数,可忽略
	void addFeature(Mat m);
	/**
	@breif 仿射变换，用于人脸矫正
	@param 原图片
	@param 输出图片
	@param 原图片特征点
	@param 目标特征点
	*/
	void AffineTran(Mat src, Mat& dst, const Point2f sr[], const Point2f ds[]);
	//测试函数，可忽略
	float getConfidence(Mat m);
	/**
	@brief 检测人脸是否为正脸
	@param 需检测的人脸图片
	*/
	bool frontface(Mat m);
	/**
	@brief 保存所有已检测到的人脸
	*/
	std::vector<Face*> face_set;
private:
	Point2f standard[5];
	const float mean_val = 127.5f;
	const float std_val = 0.0078125f;
	void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
	float* extract(Mat m);
	boost::shared_ptr<caffe::Net<float> > net;
	float calculateAngle(float * _a, float* _b);
	int featuredim = 512;
	float* normlize(float* _f);
	Mat CustomAffineTransform(const Point2f src[], const Point2f dst[]);
	boost::shared_ptr<Net<float>> ONet_;
	float* getLandmark(Mat image);
	void NormizeAlign(Mat src, Mat dst);
	CascadeClassifier cascade;
};
bool Arcface::frontface(Mat m) {
	Mat Gray;
	cvtColor(m,Gray, cv::COLOR_BGR2GRAY);
	equalizeHist(Gray,Gray);
	vector<Rect> obj;
	cascade.detectMultiScale(Gray, obj);
	if (obj.size() == 1)
		return true;
	else
		return false;
}
float Arcface::getConfidence(Mat image) {
	cv::Size origin(image.cols, image.rows);
	Blob<float>* input_layer = nullptr;
	input_layer = ONet_->input_blobs()[0];
	input_layer->Reshape(1, 3, 48, 48);
	ONet_->Reshape();
	cv::Mat re_img;
	cv::resize(image, re_img, cv::Size(48, 48), 0, 0, cv::INTER_LINEAR);
	float *input_data = input_layer->mutable_cpu_data();
	Vec3b *roi_data = (Vec3b *)re_img.data;
	CHECK_EQ(re_img.isContinuous(), true);
	for (int k = 0; k < 48 * 48; ++k) {
		input_data[k] = float((roi_data[k][0] - mean_val)*std_val);
		input_data[k + 48] = float((roi_data[k][1] - mean_val)*std_val);
		input_data[k + 2 * 48] = float((roi_data[k][2] - mean_val)*std_val);
	}
	ONet_->Forward();
	Blob<float>* confidence = ONet_->blob_by_name("prob1").get();
	Blob<float>* reg_box = ONet_->blob_by_name("conv6-2").get();
	Blob<float>* reg_landmark = ONet_->blob_by_name("conv6-3").get();
	const float* confidence_data = confidence->cpu_data();
	return confidence_data[1];
}
void Arcface::NormizeAlign(Mat src, Mat dst) {
	float* landmark = getLandmark(src);
	Point2f landPoint[5];
	for (int j = 0; j < 5; j++) {
		landPoint[j] = Point2f(landmark[2 * j], landmark[2 * j + 1]);
	}
	AffineTran(src, dst, landPoint, standard);
}
float* Arcface::getLandmark(Mat image) {
	cv::Size origin(image.cols, image.rows);
	Blob<float>* input_layer = nullptr;
	input_layer = ONet_->input_blobs()[0];
	input_layer->Reshape(1, 3, 48, 48);
	ONet_->Reshape();
	cv::Mat re_img;
	cv::resize(image, re_img, cv::Size(48, 48), 0, 0, cv::INTER_LINEAR);
	float *input_data = input_layer->mutable_cpu_data();
	Vec3b *roi_data = (Vec3b *)re_img.data;
	CHECK_EQ(re_img.isContinuous(), true);
	for (int k = 0; k < 48 * 48; ++k) {
		input_data[k] = float((roi_data[k][0] - mean_val)*std_val);
		input_data[k + 48] = float((roi_data[k][1] - mean_val)*std_val);
		input_data[k + 2 * 48] = float((roi_data[k][2] - mean_val)*std_val);
	}
	ONet_->Forward();
	Blob<float>* confidence = ONet_->blob_by_name("prob1").get();
	Blob<float>* reg_box = ONet_->blob_by_name("conv6-2").get();
	Blob<float>* reg_landmark = ONet_->blob_by_name("conv6-3").get();
	const float* confidence_data = confidence->cpu_data();
	const float* reg_data = reg_box->cpu_data();
	const float* landmark_data = reg_landmark->cpu_data();
	float* landmark = (float*)malloc(11 * sizeof(float));
	if (confidence_data[1] >= 0) {
		if (reg_landmark) {
			for (int i = 0; i < 5; ++i) {
				landmark[2 * i] = landmark_data[2 * i] * origin.width;
				landmark[2 * i + 1] = landmark_data[2 * i + 1] * origin.height;
			}
		}
	}
	landmark[10] = confidence_data[1];
	return landmark;
}
Mat Arcface::CustomAffineTransform(const Point2f src[], const Point2f dst[]) {
	Mat_<float> A = Mat(5, 3, CV_32FC1, Scalar(0));
	Mat_<float> B = Mat(5, 2, CV_32FC1, Scalar(0));
	for (int i = 0; i < 5; i++) {
		A(i, 0) = src[i].x;
		A(i, 1) = src[i].y;
		A(i, 2) = 1;
		B(i, 0) = dst[i].x;
		B(i, 1) = dst[i].y;
	}
	Mat tran = (A.t()*A).inv()*A.t()*B;
	tran = tran.t();
	return tran;
}
void Arcface::addFeature(Mat m) {
	Face* temp = new Face();
	temp->feature.push_back(normlize(extract(m)));
	face_set.push_back(temp);
}
void Arcface::init() {
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
	net.reset(new caffe::Net<float>("arcfacemodel\\face.prototxt", caffe::TEST));
	net->CopyTrainedLayersFrom("arcfacemodel\\face.caffemodel");
	ONet_.reset(new Net<float>(("model\\det3-half.prototxt"), caffe::TEST));
	ONet_->CopyTrainedLayersFrom("model\\det3-half.caffemodel");
	cascade.load("model\\haarcascade_frontalface_alt2.xml");
	standard[0] = Point2f(39.f, 40.f);
	standard[1] = Point2f(89.f, 40.f);
	standard[2] = Point2f(64.f, 64.f);
	standard[3] = Point2f(45.f, 97.f);
	standard[4] = Point2f(83.f, 97.f);
}
void Arcface::AffineTran(Mat src, Mat& dst, const Point2f sr[],const Point2f ds[]) {
	Mat transform =CustomAffineTransform(sr, ds);
	warpAffine(src, dst, transform,Size(128,128));
}
float Arcface::calculateAngle(float* _a, float* _b) {
	double innerProduct = 0;
	for (int i = 0; i < 512; i++) {
		innerProduct += _a[i] * _b[i];
	}
	double L2norm_a = 0;
	for (int i = 0; i < 512; i++) {
		 L2norm_a += _a[i] * _a[i];
	}
	L2norm_a = sqrt(L2norm_a);
	double L2norm_b = 0;
	for (int i = 0; i < 512; i++) {
		L2norm_b += _b[i] * _b[i];
	}
	L2norm_b = sqrt(L2norm_b);
	double cos = innerProduct / (L2norm_a*L2norm_b);
	float angle = acos(cos)/PI*180;
	return angle;
}
float* Arcface::extract(Mat m) {
	Blob<float>* input_layer_target = net->input_blobs()[0];
	std::vector<cv::Mat> target_channels;
	int target_width = input_layer_target->width();
	int target_height = input_layer_target->height();
	float* target_data = input_layer_target->mutable_cpu_data();
	for (int i = 0; i < input_layer_target->channels(); ++i) {
		cv::Mat channel(target_height, target_width, CV_32FC1, target_data);
		target_channels.push_back(channel);
		target_data += target_width * target_height;
	}
	Mat ex_img;
	if (m.cols > m.rows) {
		copyMakeBorder(m, ex_img, (m.cols - m.rows) / 2, (m.cols - m.rows) / 2, 0, 0,0 ,Scalar(255,255,255,255));
	}
	if (m.cols < m.rows) {
		copyMakeBorder(m, ex_img, (m.rows - m.cols) / 2, (m.rows - m.cols) / 2, 0, 0, 0, Scalar(255, 255, 255, 255));
	}
	if (m.cols = m.rows) ex_img = m;
	Mat re_img;
	resize(ex_img, re_img, Size_<int>(112, 112));
	Mat rgb_img;  
	cvtColor(re_img, rgb_img, COLOR_BGR2RGB);
	Preprocess(rgb_img, &target_channels);
	net->Forward();
	Blob<float> *output_layer;
	output_layer = net->output_blobs()[0];
	const float* result = output_layer->cpu_data();
	const int num_det = output_layer->height();
	float* feature = (float*)(malloc(sizeof(float) * 512));
	float L2_norm = 0;
	for (int i = 0; i < 512; i++) {
		feature[i] = result[i];
	}
	return feature;
}
void Arcface::Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net->input_blobs()[0];
	int num_channels_ = input_layer->channels();
	Size input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
	Mat mean_ = cv::Mat(input_geometry_, CV_32FC3, cv::Scalar(127, 127, 127));
	
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, COLOR_GRAY2BGR);
	else
		sample = img;

	
	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;


	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);


	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	
	cv::split(sample_normalized, *input_channels);
}
float Arcface::getAngle(Mat _m, Mat _n) {
	NormizeAlign(_m, _m);
	float* feature_m = extract(_m);
	NormizeAlign(_n, _n);
	float* feature_n = extract(_n);
	float angle = calculateAngle(feature_m,feature_n);
	return angle;
}
int Arcface::getId(Mat img) {
	cv::Point2f landmark[5];
	float confidence = 0;
	float* markandconf =getLandmark(img);
	for (int i = 0; i < 5; i++) {
		landmark[i] = cv::Point2f(markandconf[2 * i], markandconf[2 * i + 1]);
	}
	confidence = markandconf[10];
	if (confidence < 0.5) return -1;
	free(markandconf);
	Mat m;
	AffineTran(img, m, landmark, standard);
	if (face_set.size() == 0) {
		Face* face = new Face();
		face->feature.push_back(normlize(extract(m)));
		face->confidence = confidence;
		for (int i = 0; i < 5; i++) {
			face->landmark[i] = landmark[i];
		}
		face->faceimg.push_back(img);
		face_set.push_back(face);
		return face->faceid;
	}

	float* curFeature = normlize(extract(m));
	std::vector<const float*> set;
	for (int i = 0; i < face_set.size(); i++) {
		for (int j = 0; j < face_set[i]->feature.size();j++) {
			set.push_back(face_set[i]->feature[j]);
		}
	}
	vector<float> angle=getAllangle(set, curFeature);
	vector<float> bais(face_set.size());
	int count = 0;
	for (int i = 0; i < face_set.size(); i++) {
		bais[i] = 0;
		for (int j = 0; j < face_set[i]->feature.size(); j++) {
			bais[i]+= angle[count];
			count++;
		}
		bais[i] = bais[i]/face_set[i]->feature.size();
	}
	int category = -1;
	double min = 30;
	for (int i = 0; i < bais.size(); i++) {
		if (bais[i] < min) {
			min = bais[i];
			category = i;
		}
	}
	printf("the min angle : %f\n", min);
	if (min < 27) {
		face_set[category]->feature.push_back(curFeature);
		if (confidence > face_set[category]->confidence) {
			face_set[category]->markfeature = curFeature;
			face_set[category]->confidence = confidence;
		}
		return face_set[category]->faceid;
	}
	if (min >= 27) {
		Face* face = new Face();
		face->feature.push_back(curFeature);
		face->confidence = confidence;
		for (int i = 0; i < 5; i++) {
			face->landmark[i] = landmark[i];
		}
		face->faceimg.push_back(img);
		face_set.push_back(face);
		return face->faceid;
	}
}
float* Arcface::normlize(float* _f) {
	double norm = 0;
	for (int i = 0; i < 512; i++) {
		norm += _f[i] * _f[i];
	}
	norm = sqrt(norm);
	for (int i = 0; i < 512; i++) {
		_f[i] = _f[i] / norm;
	}
	return _f;
}
int Arcface::Face::count = 0;

//测试函数，可忽略
void printAngle() {
	Arcface facediff;
	facediff.init();
	vector<Mat> face;

	for (int i = 0; i <= 410; i++) {
		string path = "face\\classify\\5\\" + to_string(i) + ".jpg";
		Mat m = imread(path);
		if (m.empty()) continue;
		face.push_back(m);
	}
	printf("the size of face : %d\n", face.size());
	float sum=0;
	for (int i = 0; i <face.size(); i++) {
		sum += facediff.getAngle(face[18], face[i]);
		printf("face : %d angle : %f\n", i, facediff.getAngle(face[410], face[i]));
	}
	printf("average : %f\n", sum / (face.size() - 1));
}
//测试函数，可忽略
void classify() {
	Arcface facediff;
	facediff.init();
	vector<Mat> face;
	for (int i = 0; i <= 8974; i++) {
		string path = "face\\face\\" + to_string(i) + ".jpg";
		Mat m = imread(path);
		if (m.empty()) continue;
		face.push_back(m);
	}
	printf("face count : %d\n", face.size());
	vector<vector<Mat>*> classfication;

	for (int i = 0; i < face.size(); i++) {
		printf("face : %d\n", i);
		bool flag = false;
		float min = 180;
		int category = 0;
		printf("number of category : %d\n", classfication.size());
		if (!facediff.frontface(face[i]))
			continue;
		if (facediff.getConfidence(face[i]) < 0.6) continue;
		for (int j = 0; j < classfication.size(); j++) {
			float angle = facediff.getAngle(face[i], (*classfication[j])[0]);
			if (angle < min) {
				min = angle;
				category = j;
			}
		}
		if (min < 30) {
			classfication[category]->push_back(face[i]);
			flag = true;
			continue;
		}
		if (!flag) {
			vector<Mat>* temp = new vector<Mat>();
			temp->push_back(face[i]);
			classfication.push_back(temp);
		}
	}
	printf("category : %d\n", classfication.size());
	for (int i = 0; i < classfication.size(); i++) {
		printf("the number of face category %d is %d\n", i, classfication[i]->size());
	}
	for (int i = 0; i < classfication.size(); i++) {
		for (int j = 0; j < classfication[i]->size(); j++) {
			string path = "face\\classify\\" + to_string(i) + "\\" + to_string(j) + ".jpg";
			//imwrite(path, (*classfication[i])[j]);
		}
	}
}

int main() {
	//printAngle();
	//classify();
	
	Arcface facediff;
	facediff.init();
	vector<Mat> face;
	for (int i = 0; i <= 8974; i++) {
		string path = "face\\face\\" + to_string(i) + ".jpg";
		Mat m = imread(path);
		if (m.empty()) continue;
		face.push_back(m);
	}
	printf("face count : %d\n", face.size());

	vector<vector<Mat>*> classify;
	for (int i = 0; i < face.size(); i++) {
		if (!facediff.frontface(face[i]))
			continue;
		clock_t start=clock();
		int id = facediff.getId(face[i]);
		if (id == -1) continue;
		clock_t end=clock();
		printf("face : %d category : %d the used time : %f\n", i,classify.size(),(end-start)/CLOCKS_PER_SEC*1000);
		if (id >= classify.size()) {
			vector<Mat>* temp = new vector<Mat>();
			temp->push_back(face[i]);
			classify.push_back(temp);
		}else {
			classify[id]->push_back(face[i]);
		}
	}
	printf("categoy : %d\n", classify.size());
	for (int i = 0; i < classify.size(); i++) {
		printf("%d  :  %d\n", i, classify[i]->size());
	}
	for (int i = 0; i < classify.size(); i++) {
		for (int j = 0; j < classify[i]->size(); j++) {
			string path = "face\\classify\\" + to_string(i) + "\\" + to_string(j) + ".jpg";
			imwrite(path, (*classify[i])[j]);
		}
	}
	for (int i = 0; i < facediff.face_set.size(); i++) {
		printf("category of %d is %d\n", i, facediff.face_set[i]->feature.size());
	}
	int a = 0;
	scanf_s("%d", a);
}
