#ifndef Scene_Text_Recognizer_H
#define Scene_Text_Recognizer_H

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/strings/str_util.h"
//#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/reporter.h"

//opencv
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"

#include "utils.h"

using namespace tensorflow;


class SceneTextRecognizer{
  public:
    SceneTextRecognizer();

    SceneTextRecognizer(std::string frozen_graph_filename, std::string dictionary_filename, int _im_height=32, int _im_width=128);
    
    bool init(const std::string frozen_graph_filename,const std::string);
    void preprocess_image(cv::Mat& input_image, cv::Mat& output_image);
    std::vector<cv::Mat> preprocess_image(std::vector<cv::Mat>& input_images);
    std::string run_graph(const cv::Mat& image);
    std::vector<std::string> run_graph(const std::vector<cv::Mat> input_images);
    bool init_graph(const std::string&);
    bool init_dictionary(const std::string&);
    std::string decode_single_text(std::vector<int>& vec);

  private:
    void init_constant_vars(int _im_height=32, int _im_width=128);
    tensorflow::GraphDef graph_def;
    std::vector<string> input_layers;
    std::unique_ptr<tensorflow::Session> session;
    std::vector<string> output_layers;
    float width_scale_ratio;
    int seq_len;
    int image_width;
    int image_height;
    std::unordered_map<int, char> mapping;
};

//for debug purpose
#endif 
