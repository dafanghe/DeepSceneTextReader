#ifndef Scene_Text_Reader_H
#define Scene_Text_Reader_H

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <assert.h>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

//opencv
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"

#include "utils.h"
//recognizer
#include "ctc_scene_text_recognizer.h"
#include "recognizer.h"
//detector
#include "faster_rcnn_text_detector.h"
#include "detector.h"

#include "text_box.h"

using namespace tensorflow;

namespace scene_text_reader{

  class SceneTextReader{
    public:
      SceneTextReader();

      SceneTextReader(const std::string&, const std::string&, const std::string&,
                      const std::string& detector_model=std::string("FasterRCNN"),
                      const std::string& recognizer_model=std::string("CTC"));
    
      void read_text(cv::Mat&, std::vector<TextBox>& res);

      void extract_word_regions(cv::Mat& image,
          std::vector<TextBox>& boxes, std::vector<cv::Mat>& word_regions);
    
    private:
      Detector *detector;
      Recognizer *recognizer; 
  };

}
#endif 
