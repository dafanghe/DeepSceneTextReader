#include "text_box.h"

TextBox::TextBox(std::vector<cv::Point>& points, std::string text){
  this->points = points;
  this->text = text;
}

void TextBox::get_rectangle_box(cv::Point& p1, cv::Point& p2){
  int minx = 100000, miny = 100000, maxx = 0, maxy = 0;
  for(auto const& value: this->points){
    minx = std::min(minx, value.x);
    miny = std::min(miny, value.y);
    maxx = std::max(maxx, value.x);
    maxy = std::max(maxy, value.y);
  }
  p1.x = minx;
  p1.y = miny;
  p2.x = maxx;
  p2.y = maxy;
}

std::ostream &operator<<(std::ostream &os, TextBox &m) { 
  std::vector<cv::Point> points = m.get_points();
  os<<"oriented box: ";
  for(int i = 0; i < points.size(); i++){
    os<<points[i]<<" ";
  }
  os<<" text: "<<m.get_text()<<std::endl;
  return os;
}

