#ifndef Recognizer_H
#define Recognizer_H

#include <vector>
#include <unordered_map>
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


class Recognizer{
  //A base class should implemented the following functions:
  //Preprocess_image: preprocess a single image represented as an opencv mat
  //Preprocess images: preprocess a vector of opencv mat images
  public:
    Recognizer(){};
    Recognizer(const std::string& recognizer_graph_filename, const std::string& dictionary_filename){
      init_dictionary(dictionary_filename);
      init_graph(recognizer_graph_filename);
    };
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
    bool init_dictionary(const std::string& filename){
      std::ifstream inf(filename, std::ios::in);
      if(!inf.is_open())
      { LOG(ERROR)<<"Error dictionary opening file "<<filename; std::exit(1); }

      LOG(INFO) <<"read dictionary file "<<filename;
      std::string line;
      std::vector<std::string> splits;
      while(!inf.eof()){
        inf>>line;
        splits = tensorflow::str_util::Split(line, ',');
        this->mapping[std::stoi(splits[0])] = splits[1][0];
      }
      inf.close();
      return 1;
    }
    virtual void preprocess_image(cv::Mat& input_image, cv::Mat& output_image) = 0;
    virtual std::vector<cv::Mat> preprocess_images(std::vector<cv::Mat>& input_images) = 0;
    virtual std::string run_graph(const cv::Mat& image) = 0;
    virtual std::vector<std::string> run_graph(const std::vector<cv::Mat> input_images) = 0;
    std::string decode_single_text(std::vector<int>& vec){
      std::string res;
      for(int i=0; i<vec.size(); i++){
        res.push_back(this->mapping[vec[i]]);
      }
      return res;
    }
    std::unordered_map<int, char> mapping;
    tensorflow::GraphDef graph_def;
    std::unique_ptr<tensorflow::Session> session;
    std::vector<std::string> input_layers;
    std::vector<std::string> output_layers;
};

#endif
