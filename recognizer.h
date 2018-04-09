#ifndef Recognizer_H
#define Recognizer_H

#include <vector>
#include <unordered_map>
#include <string>

//opencv
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/graph.pb.h"


class Recognizer{
  //A base class should implemented the following functions:
  //Preprocess_image: preprocess a single image represented as an opencv mat
  //Preprocess images: preprocess a vector of opencv mat images
  public:
    virtual void preprocess_image(cv::Mat& input_image, cv::Mat& output_image) = 0;
    virtual std::vector<cv::Mat> preprocess_images(std::vector<cv::Mat>& input_images) = 0;
    virtual std::string run_graph(const cv::Mat& image) = 0;
    virtual std::vector<std::string> run_graph(const std::vector<cv::Mat> input_images) = 0;
    std::string decode_single_text(std::vector<int>& vec){
      std::string res;
      for(int i=0; i<vec.size(); i++){
        res.push_back(this->mapping[vec[i]]);
      }
      return res;
    }
    std::unordered_map<int, char> mapping;
    tensorflow::GraphDef graph_def;
    std::unique_ptr<tensorflow::Session> session;
    std::vector<std::string> input_layers;
    std::vector<std::string> output_layers;
};

#endif
