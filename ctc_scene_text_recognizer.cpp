#include "ctc_scene_text_recognizer.h"


CTCSceneTextRecognizer::CTCSceneTextRecognizer(){
  init_constant_vars(); 
}


CTCSceneTextRecognizer::CTCSceneTextRecognizer(std::string frozen_graph_filename, std::string dictionary_filename, int _im_height, int _im_width):Recognizer(frozen_graph_filename, dictionary_filename){
  init_constant_vars(_im_height, _im_width);
}


bool CTCSceneTextRecognizer::init(const std::string frozen_graph_filename, const std::string dictionary_filename){
  this->init_graph(frozen_graph_filename); 
  this->init_dictionary(dictionary_filename);
  return true;
}


void CTCSceneTextRecognizer::init_constant_vars(int _im_height, int _im_width){
  std::string input_layer_string = "input_images:0,input_seq_lens:0";
  std::string output_layer_string = "CTCBeamSearchDecoder:0,CTCBeamSearchDecoder:1,CTCBeamSearchDecoder:2";
  this->input_layers = str_util::Split(input_layer_string, ',');
  this->output_layers = str_util::Split(output_layer_string, ',');
  this->seq_len = 29;
  this->image_width = _im_width;  //input image width;
  this->image_height = _im_height;  //input image height
  this->width_scale_ratio = 1.2; //scale the width for better recognition
}

    
void CTCSceneTextRecognizer::preprocess_image(cv::Mat& input_image, cv::Mat& output_image){
  cv::Mat resized_image, padded_image;
  int new_width = int(this->width_scale_ratio * input_image.cols);
  cv::resize(input_image, input_image, cv::Size(new_width, input_image.rows));
  float ratio=0;
  resize_image_fix_height(input_image, resized_image, ratio, this->image_height);
  pad_image_width(resized_image, output_image, this->image_width);
}


string CTCSceneTextRecognizer::run_graph(const cv::Mat& image){
  int height = image.rows;
  int width = image.cols;
  Tensor input_img_tensor(DT_FLOAT, TensorShape({1, height, width, 3}));

  unsigned char *input_data = (unsigned char*)(image.data); 
  auto input_tensor_mapped = input_img_tensor.tensor<float, 4>();
  //(TODO) is there any other ways to copy the data into tensor?
  for (int y = 0;y < height; ++y) {
    for (int x = 0;x < width; ++x) {
      unsigned char b = input_data[image.step * y + x * image.channels()];
      unsigned char g = input_data[image.step * y + x * image.channels() + 1];
      unsigned char r = input_data[image.step * y + x * image.channels() + 2];
      input_tensor_mapped(0, y, x, 0) = float(r);
      input_tensor_mapped(0, y, x, 1) = float(g);
      input_tensor_mapped(0, y, x, 2) = float(b);
    }
  }
  //create the seq len tensor and assign fixed value
  Tensor input_seq_len_tensor(DT_INT32, TensorShape({1}));
  auto input_seq_len_mapped = input_seq_len_tensor.tensor<int, 1>();
  input_seq_len_mapped(0) = this->seq_len;

  //create the input to run
  std::vector<std::pair<string, Tensor> > inputs = {
    {this->input_layers[0], input_img_tensor}, 
    {this->input_layers[1], input_seq_len_tensor},
  };

  std::vector<Tensor> outputs;
  Status run_status = this->session->Run(inputs,
            this->output_layers, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return "";
  }
  LOG(INFO) <<"number of output:"<<outputs.size();
  
  auto indices = outputs[0].flat_outer_dims<long long>();
  auto values = outputs[1].flat_outer_dims<long long>();
  
  const Eigen::Tensor<float, indices.NumDimensions>::Dimensions& indices_dim = indices.dimensions();
  const Eigen::Tensor<float, values.NumDimensions>::Dimensions& values_dim = values.dimensions();
 
  LOG(INFO) << outputs[0].DebugString();
  LOG(INFO) << outputs[1].DebugString();
  std::vector<int> encoded_text;
  for(int i=0; i<values_dim[0]; i++){
    for(int j=0; j<values_dim[1]; j++){
      encoded_text.push_back(values(i,j));
    }
  }
  std::string decoded_text = decode_single_text(encoded_text); 
  return decoded_text;
}
    

std::vector<cv::Mat> CTCSceneTextRecognizer::preprocess_images(std::vector<cv::Mat>& input_images){
  std::vector<cv::Mat> processed_images(input_images.size());
  for(int i=0; i<input_images.size(); i++){
    cv::Mat preprocessed_image;
    this->preprocess_image(input_images[i], preprocessed_image);
    processed_images[i] = preprocessed_image;
  }
  return processed_images;
}
    

std::vector<std::string> CTCSceneTextRecognizer::run_graph(const std::vector<cv::Mat> images){
  //the images must be preprocessd and has the same height and width!!
  std::vector<std::string> res;
  int num_word = images.size();
  if(num_word == 0) return res;

  int height = this->image_height;
  int width = this->image_width;
  Tensor input_img_tensor(DT_FLOAT, TensorShape({num_word, height, width, 3}));
  auto input_tensor_mapped = input_img_tensor.tensor<float, 4>();
  //create the seq len tensor and assign fixed value for ctc
  Tensor input_seq_len_tensor(DT_INT32, TensorShape({num_word}));
  auto input_seq_len_mapped = input_seq_len_tensor.tensor<int, 1>();

  for(int i=0; i<num_word; i++){
    const cv::Mat& image = images[i];
    //std::cout<<"assign image to tensor"<<i<<" "<<image.rows<<" "<<image.cols<<std::endl;
    assert (image.rows == height);
    assert (image.cols == width);
    const unsigned char *input_data = (const unsigned char*)(image.data); 
    //(TODO) is there any other ways to copy the data into tensor?
    for (int y = 0;y < height; ++y) {
      for (int x = 0;x < width; ++x) {
        const unsigned char b = input_data[image.step * y + x * image.channels()];
        const unsigned char g = input_data[image.step * y + x * image.channels() + 1];
        const unsigned char r = input_data[image.step * y + x * image.channels() + 2];
        input_tensor_mapped(i, y, x, 0) = float(r);
        input_tensor_mapped(i, y, x, 1) = float(g);
        input_tensor_mapped(i, y, x, 2) = float(b);
      }
    }
    input_seq_len_mapped(i) = this->seq_len;
  }
  //create the input to run
  std::vector<std::pair<string, Tensor> > inputs = {
    {this->input_layers[0], input_img_tensor}, 
    {this->input_layers[1], input_seq_len_tensor},
  };

  //std::cout<<"run recognition graph"<<std::endl;
  std::vector<Tensor> outputs;
  Status run_status = this->session->Run(inputs,
            this->output_layers, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return res;
  }
  LOG(INFO) <<"number of output:"<<outputs.size();
 
  //std::cout<<outputs[0].DebugString()<<std::endl;
  //std::cout<<outputs[1].DebugString()<<std::endl;
  auto indices_shape = outputs[0].shape();

  auto indices = outputs[0].tensor<long long, 2>();
  auto values = outputs[1].tensor<long long, 1>();
  
  //const Eigen::Tensor<float, indices.NumDimensions>::Dimensions& indices_dim = indices.dimensions();
  //const Eigen::Tensor<float, values.NumDimensions>::Dimensions& values_dim = values.dimensions();

  std::vector<std::vector<int> > encoded_texts(num_word);
  for(int i=0; i<indices_shape.dim_size(0); i++){
    encoded_texts[indices(i, 0)].push_back(values(i));
  }
  
  for(int i=0; i<num_word; i++){
    res.push_back(decode_single_text(encoded_texts[i]));
  }
  return res;
}
