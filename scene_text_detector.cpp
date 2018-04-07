#include "scene_text_detector.h"

void visualize_det_output(std::vector<Tensor>& outputs){
  std::cout<<outputs[0].DebugString()<<std::endl;
  std::cout<<outputs[1].DebugString()<<std::endl;

  int dim = outputs[0].dims();
  std::cout<<"num dim: "<<dim<<" "<<outputs[0].NumElements()<<std::endl;
  
  //auto out_score_mat = outputs[0].matrix<float>();
  //std::cout<<out_score_mat.NumDimensions<<std::endl;

  //const Eigen::Tensor<float, out_score.NumDimensions>::Dimensions& out_score_dim = out_score.dimensions();
  //LOG(INFO)<<"score dimensions: "<<out_score.NumDimensions<<" "<<out_score_dim[0];

  //tensor to cv mat
  cv::Mat vis_score = tensor_to_cv_mat(outputs[0]);
  //std::cout<<vis_score.rows<<" "<<vis_score.cols<<std::endl;
  double min_v, max_v;
  cv::minMaxLoc(vis_score, &min_v, &max_v);
  //std::cout<<"max score: "<<max_v<<" min score "<<min_v<<std::endl;

  //cv::imwrite("vis_score.jpg", vis_score);
  //visualize geometry
  auto out_geometry = outputs[1].flat_outer_dims<float>();
  //LOG(INFO)<<"geometry dimensions: "<<out_geometry.NumDimensions;
}

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


int SceneTextDetector::run_graph(const std::string image_filename){
  std::vector<Tensor> inputs;
  if (!ReadTensorFromImageFile(image_filename, &inputs).ok()) {
    LOG(ERROR) << "Load image";
    return -1;
  }
  std::vector<Tensor> outputs;
  
  const Tensor& image_tensor = inputs[0];
  Status run_status = session->Run({{input_layer, image_tensor}},
                                   output_layers, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }
  LOG(INFO) <<"number of output:"<<outputs.size();
  //auto score = outputs[0].flat_outer_dims<float>();
  //auto geometry = outputs[1].flat_outer_dims<float();

  //LOG(INFO)<<"geometry dimensions: "<<geometry.NumDimensions;
//  Eigen::Tensor<float, 2>& vec = out_score.flat();
//  
//  auto scores = outputs[1].flat<float>();
//  tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
//  auto num_detections = outputs[4].flat<float>();
//
//  LOG(INFO) << "rank of boxes:" << boxes.NumDimensions;
//  const Eigen::Tensor<float, boxes.NumDimensions>::Dimensions& dim = boxes.dimensions();
//  //LOG(INFO) << "Dim size: " << dim.size << ", dim 0: " << dim[0]
//  //  << ", dim 1: " << dim[1];
//  
//  LOG(INFO) << "rank of oriented boxes:" << oriented_boxes.NumDimensions;
//  const Eigen::Tensor<float, oriented_boxes.NumDimensions>::Dimensions& odim = oriented_boxes.dimensions();
//  //LOG(INFO) << "Dim size: " << odim.size << ", dim 0: " << odim[0]
//  //   << ", dim 1: " << odim[1];
//
//  LOG(INFO) << "number of detections:"<<num_detections(0) <<", "<< outputs[0].shape().DebugString();
//  for(size_t i = 0; i < num_detections(0) && i < 20;++i)
//  {
//    if(scores(i) > 0.5)
//    {
//      LOG(INFO)<<i<<"***** score: "<< scores(i) << ", class: ********" << classes(i);
//      LOG(INFO)<<" box:" << boxes(0, i*4) << ","<< boxes(0, i*4+1) <<"," << boxes(0, i*4+2)<<","<<boxes(0, i*4+3);
//      LOG(INFO)<<" oriented box:";
//      for(size_t j=0; j<4; j++){
//        LOG(INFO) << oriented_boxes(0, i*8+2*j) << ","<< oriented_boxes(0, i*8+2*j+1);
//      }
//    }
//  }
  return 0;
}
    
Status SceneTextDetector::ReadTensorFromImageFile(string file_name, std::vector<Tensor>* out_tensors) {
  //auto root = tensorflow::Scope::NewRootScope();
//g  auto root = ::Scope::NewRootScope();
//g  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
//g
//g  string input_name = "file_reader";
//g  auto file_reader = tensorflow::ops::ReadFile(root.WithOpName(input_name),
//g                                               file_name);
//g  // Now try to figure out what kind of file it is and decode it.
//g  const int wanted_channels = 3;
//g  tensorflow::Output image_reader;
//g  // it's a JPEG.
//g  image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
//g                            DecodeJpeg::Channels(wanted_channels));
//g  auto uint8_caster =  Cast(root.WithOpName("uint8_caster"), image_reader, tensorflow::DT_UINT8);
//g  auto dims_expander = ExpandDims(root.WithOpName("dim"), uint8_caster, 0);
//g
//g  // This runs the GraphDef network definition that we've just constructed, and
//g  // returns the results in the output tensor.
//g  tensorflow::GraphDef graph;
//g  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
//g
//g  std::unique_ptr<tensorflow::Session> session(
//g      tensorflow::NewSession(tensorflow::SessionOptions()));
//g  TF_RETURN_IF_ERROR(session->Create(graph));
//g  TF_RETURN_IF_ERROR(session->Run({}, {"dim"}, {}, out_tensors));
  return Status::OK();
  //return 1;
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
