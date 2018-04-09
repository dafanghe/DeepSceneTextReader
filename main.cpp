#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "scene_text_detector.h"
#include "ctc_scene_text_recognizer.h"
#include "scene_text_reader.h"
#include "utils.h"

int detect_text(string& detector_graph_filename, string& image_filename, string& output_filename)
{
  LOG(INFO)<<"start text detection:";

  SceneTextDetector detector(detector_graph_filename);

  cv::Mat image = cv::imread(image_filename);
  if(!image.data)                              // Check for invalid input
  {
      LOG(ERROR) <<  "Could not open or find the image " << image_filename;
      return -1;
  } 
  std::vector<cv::Scalar> colors={cv::Scalar(0,0,255), cv::Scalar(0,255,0),
    cv::Scalar(255,0,0), cv::Scalar(255,255,0), cv::Scalar(0,255,255), cv::Scalar(255,0,255)};
  std::vector<TextBox> res;
  detector.run_graph(image, res);
  for(int i=0; i<res.size(); i++){
    std::vector<cv::Point> points = res[i].get_points(); 
    for(int j=0; j<4; j++){
      cv::line(image, points[j], points[(j+1)%4], colors[j%4], 3); 
    }
  }
  
  //write image
  cv::imwrite(output_filename, image);
  return 0;
}


int recognize_text(string& recognizer_graph_filename, string& dictionary_filename,
    string& image_filename, int im_height=32, int im_width=128)
{
  LOG(INFO) <<"start text recognition: "<<recognizer_graph_filename;
  CTCSceneTextRecognizer recognizer(recognizer_graph_filename, dictionary_filename, im_height, im_width);
  cv::Mat image = cv::imread(image_filename);
  if(!image.data)                              // Check for invalid input
  {
    LOG(ERROR) <<  "Could not open or find the image " << image_filename;
    return -1;
  }
  LOG(INFO)<<" read text image: "<<image.rows<<" "<<image.cols;
  cv::Mat preprocessed_image;
  recognizer.preprocess_image(image, preprocessed_image);
  string res = recognizer.run_graph(preprocessed_image);
  LOG(INFO)<<"prediction : "<<res;
  return 0;
}


int end_to_end_reading(string& detector_graph_filename, string& recognizer_graph_filename,
    string& dictionary_filename, string& image_filename, string& output_filename)
{
  scene_text_reader::SceneTextReader reader(detector_graph_filename,
      recognizer_graph_filename, dictionary_filename); 

  cv::Mat image = cv::imread(image_filename);
  if(!image.data)                              // Check for invalid input
  {
      LOG(ERROR) <<  "Could not open or find the image " << image_filename;
      return -1;
  } 
  std::vector<TextBox> res;
  reader.read_text(image, res);
  for(int i=0; i<res.size(); i++){
    std::cout<<res[i];
    //draw_polygon(image, res[i].get_points());
    draw_text_box(image, res[i]);
  } 
  cv::imwrite(output_filename, image);
}


int main(int argc, char** argv) {
  //do text detection
  string detector_graph = "";
  string recognizer_graph = "";
  string dictionary_filename = "";
  string image_filename = "";
  string output_filename = "";
  int im_height, im_width;
  string mode = "";
  std::vector<Flag> flag_list = {
    Flag("detector_graph", &detector_graph, "detector graph file name"),
    Flag("recognizer_graph", &recognizer_graph, "recognizer graph file name"),
    Flag("im_height", &im_height, "image height for recognition model"),
    Flag("im_width", &im_width, "image width for recognition model"),
    Flag("dictionary_filename", &dictionary_filename, "dictionary filename for decode the recognition results"),
    Flag("image_filename", &image_filename, "the filename to be tested."),
    Flag("output_filename", &output_filename, "the output filename"),
    Flag("mode", &mode, "the mode, must be within the three categories: detect, recognize, detect_and_read"),
  };

  string usage = Flags::Usage(argv[0], flag_list);
  const bool parse_result = Flags::Parse(&argc, argv, flag_list);

  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  ::tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }
  
  if(mode == "detect"){
    detect_text(detector_graph, image_filename, output_filename);
  }else if(mode == "recognize"){
    recognize_text(recognizer_graph, dictionary_filename, image_filename, im_height, im_width);
  }else if(mode == "detect_and_read"){
    end_to_end_reading(detector_graph, recognizer_graph,
      dictionary_filename, image_filename, output_filename);
  }else{
    LOG(ERROR) << "mode should be within: detect, recognize, detect_and_read";
  }
}
