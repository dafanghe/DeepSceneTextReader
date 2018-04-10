#ifndef Detector_H
#define Detector_H

#include <vector>
#include <string>

//opencv
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"

//tensorflow
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/lib/strings/str_util.h"

#include "text_box.h"


class Detector{
  public:
    Detector(){};
    Detector(const std::string frozen_graph_filename){
      init_graph(frozen_graph_filename);
    }
    bool init_graph(const std::string& frozen_graph_filename){
      if (!ReadBinaryProto(tensorflow::Env::Default(), frozen_graph_filename, &graph_def).ok()) {
        LOG(ERROR) << "Read proto";
        return -1;
      } 
      
      tensorflow::SessionOptions sess_opt;
      sess_opt.config.mutable_gpu_options()->set_allow_growth(true);
      (&session)->reset(tensorflow::NewSession(sess_opt));
      if (!session->Create(graph_def).ok()) {
        LOG(ERROR) << "Create graph";
        return -1;
      }
    }
    virtual int run_graph(const cv::Mat& image, std::vector<TextBox>& results) = 0;

    tensorflow::GraphDef graph_def;
    std::string input_layer; //for detector, we assume there is only one input
    std::unique_ptr<tensorflow::Session> session;
    std::vector<std::string> output_layers;
};

#endif
