#ifndef FasterRCNN_Text_Detector_H
#define FasterRCNN_Text_Detector_H

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

#include "detector.h"
#include "text_box.h"
#include "utils.h"

using namespace tensorflow;


class FasterRCNNTextDetector: public Detector{
  public:
    FasterRCNNTextDetector(){};

    FasterRCNNTextDetector(const std::string frozen_graph_filename);
    
    bool init_constants();
    int run_graph(const cv::Mat& image, std::vector<TextBox>& results);

  private:
    float score_thresh;
};

#endif 
