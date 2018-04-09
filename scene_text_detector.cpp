#include "scene_text_detector.h"


SceneTextDetector::SceneTextDetector(const std::string frozen_graph_filename){
  this->init(frozen_graph_filename);
}

bool SceneTextDetector::init(const std::string frozen_graph_filename){
  init_graph(frozen_graph_filename); 
  input_layer = "image_tensor:0";
  output_layers = str_util::Split("detection_boxes:0,detection_scores:0,detection_classes:0,detection_oriented_boxes:0,num_detections:0", ',');
  score_thresh = 0.6;
}

int SceneTextDetector::run_graph(const cv::Mat& image, std::vector<TextBox>& results){
  cv::Mat resized_image;
  float ratio_h=0, ratio_w=0;
  resize_image_max_len(image, resized_image, ratio_h, ratio_w);
  
  auto input_tensor = cv_mat_to_tensor(resized_image);

  std::vector<Tensor> outputs;
  Status run_status = this->session->Run({{this->input_layer, input_tensor}},
                                   this->output_layers, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }
  LOG(INFO) <<"number of output:"<<outputs.size();

  //cv::imwrite("test0.jpg", image);
  //output_layers = str_util::Split("detection_boxes:0,detection_scores:0,detection_classes:0,detection_oriented_boxes:0,num_detections:0", ',');
  auto detection_boxes = outputs[0].tensor<float, 3>();
  auto detection_scores = outputs[1].tensor<float, 2>();
  auto detection_classes = outputs[2].tensor<float, 2>();
  auto detection_oriented_boxes = outputs[3].tensor<float, 4>();

  int num_box = detection_boxes.dimension(1);
  for(int i=0;i<num_box;i++){
    if(detection_scores(0, i) > this->score_thresh){
      std::vector<cv::Point> points;
      for(int j=0; j<4; j++){
        cv::Point p;
        p.x = int(detection_oriented_boxes(0, i, j, 1) * image.cols);
        p.y = int(detection_oriented_boxes(0, i, j, 0) * image.rows);
        points.push_back(p);
      }
      TextBox tb(points, "");
      results.push_back(tb);
    }
  }
}


bool SceneTextDetector::init_graph(const std::string& frozen_graph_filename){
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
