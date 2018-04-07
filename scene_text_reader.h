#ifndef Scene_Text_Reader_H
#define Scene_Text_Reader_H

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <assert.h>

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
#include "scene_text_recognizer.h"
#include "scene_text_detector.h"
#include "text_box.h"

using namespace tensorflow;

namespace scene_text_reader{

  class SceneTextReader{
    public:
      SceneTextReader();

      SceneTextReader(std::string&, std::string&, std::string&);
    
      void read_text(cv::Mat&, std::vector<TextBox>& res);

      void extract_word_regions(cv::Mat& image,
          std::vector<TextBox>& boxes, std::vector<cv::Mat>& word_regions);
    
//      void create_recognition_input(std::vector<cv::Mat>& word_regions, std::vector<cv::Mat>& inputs);
    private:
      SceneTextDetector detector;
      SceneTextRecognizer recognizer;
  };

}
#endif 
