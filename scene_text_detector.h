#ifndef Scene_Text_Detector_H
#define Scene_Text_Detector_H

#include <iostream>
#include <memory>
#include <string>
#include <vector>

//tensorflow
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

//opencv
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"

#include "text_box.h"
#include "utils.h"

using namespace tensorflow;


class SceneTextDetector{
  public:
    SceneTextDetector(){};

    SceneTextDetector(const std::string frozen_graph_filename);
    //std::string input_layer_, std::string output_layer_string);
    
    bool init(const std::string);
    int run_graph(std::string image_filename);
    int run_graph(const cv::Mat& image, std::vector<TextBox>& results);

  private:
    Status ReadTensorFromImageFile(string file_name, std::vector<Tensor>* out_tensors);

    bool init_graph(const std::string& frozen_graph_filename);
    tensorflow::GraphDef graph_def;
    std::string input_layer;
    std::unique_ptr<tensorflow::Session> session;
    std::vector<string> output_layers;
    float score_thresh;
};

void visualize_det_output(std::vector<Tensor>&);
#endif 
