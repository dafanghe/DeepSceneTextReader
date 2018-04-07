#ifndef Text_Box_H
#define Text_Box_H

#include <iostream>
#include <memory>
#include <string>
#include <vector>

//opencv
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"

class TextBox{
  public:
    TextBox(std::vector<cv::Point>& points, std::string text);
    void get_rectangle_box(cv::Point& p1, cv::Point& p2);
    std::vector<cv::Point>& get_points() {return points;}
    std::string& get_text() {return text;}
    float& get_score() {return score;}
    void set_text(std::string s) {text=s;}
  private:
    std::vector<cv::Point> points;
    std::string text;
    float score;
};

std::ostream &operator<<(std::ostream &os, TextBox &m); 

#endif 
