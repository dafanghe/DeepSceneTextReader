#ifndef CTC_Scene_Text_Recognizer_H
#define CTC_Scene_Text_Recognizer_H

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"

//opencv
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"

#include "utils.h"

#include "recognizer.h"

using namespace tensorflow;


class CTCSceneTextRecognizer: public Recognizer{
  public:
    CTCSceneTextRecognizer();

    CTCSceneTextRecognizer(std::string frozen_graph_filename, std::string dictionary_filename, int _im_height=32, int _im_width=128);
    
    bool init(const std::string frozen_graph_filename,const std::string);
    void preprocess_image(cv::Mat& input_image, cv::Mat& output_image);
    std::vector<cv::Mat> preprocess_images(std::vector<cv::Mat>& input_images);
    std::string run_graph(const cv::Mat& image);
    std::vector<std::string> run_graph(const std::vector<cv::Mat> input_images);
    bool init_graph(const std::string&);
    bool init_dictionary(const std::string&);

  private:
    void init_constant_vars(int _im_height=32, int _im_width=128);
    float width_scale_ratio;
    int seq_len;
    int image_width;
    int image_height;
};

//for debug purpose
#endif 
