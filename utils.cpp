#include "utils.h"

#define PI 3.14159265358979323846

void resize_image_max_len(const cv::Mat& image, cv::Mat& resized_image, float& ratio_h, float& ratio_w, int max_side_len){
  int height = image.rows;
  int width = image.cols;
  float ratio = 1;
  if(std::max(height, width) > max_side_len)
    ratio = height > width ? float(max_side_len)/height: float(max_side_len)/width;
  int resize_h = int(height * ratio);
  int resize_w = int(width * ratio);
  resize_h = resize_h%32 == 0? resize_h : (resize_h/32 - 1) * 32;
  resize_w = resize_w%32 == 0? resize_w : (resize_w/32 - 1) * 32;
  cv::resize(image, resized_image, cv::Size(resize_w, resize_h));
  
  ratio_h = float(resize_h)/height;
  ratio_w = float(resize_w)/width;
}

void resize_image_fix_height(const cv::Mat& image, cv::Mat& resized_image, float& ratio, int fixed_height){
  int height = image.rows;
  int width = image.cols;
  ratio = float(fixed_height)/height;
  int resize_h = fixed_height;
  int resize_w = int(width * ratio);
  cv::resize(image, resized_image, cv::Size(resize_w, resize_h));  
}

void pad_image_width(const cv::Mat& image, cv::Mat& padded_image, int target_width){
  int height = image.rows;
  int width = image.cols;
  int borderType = cv::BORDER_CONSTANT;
  if(width > target_width)
    cv::resize(image, padded_image, cv::Size(target_width, height));
  else if(width < target_width){
    int pad_len = target_width - width;
    copyMakeBorder(image, padded_image, 0, 0, 0, pad_len, borderType, cv::Scalar(0,0,0));
  }else
    padded_image = image.clone();
}

tensorflow::Tensor cv_mat_to_tensor(const cv::Mat& image){
  int height = image.rows;
  int width = image.cols;
  int depth = 3;
  tensorflow::Tensor res_tensor(tensorflow::DT_UINT8, tensorflow::TensorShape({1, height, width, 3}));

  //we assume that the image is unsigned char dtype
  const unsigned char *source_data = (unsigned char*)(image.data); 

  auto tensor_mapped = res_tensor.tensor<unsigned char, 4>();
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      auto b = source_data[image.step * y + x * image.channels()];
      auto g = source_data[image.step * y + x * image.channels()+1];
      auto r = source_data[image.step * y + x * image.channels()+2];
      tensor_mapped(0, y, x, 0) = r;
      tensor_mapped(0, y, x, 1) = g;
      tensor_mapped(0, y, x, 2) = b;
    }
  }
  return res_tensor;
}

cv::Mat tensor_to_cv_mat(const tensorflow::Tensor tensor){
  auto tensor_data = tensor.flat<float>();
  //assume it is a 4d tensor
  auto tensor_shape = tensor.shape();
  int height = tensor_shape.dim_size(1);
  int width = tensor_shape.dim_size(2);
  std::cout<<" height "<<height << " width "<< width<<std::endl;

  cv::Mat res_mat = cv::Mat(height, width, CV_32FC1, cv::Scalar(0));
  float *res_data = (float*)(res_mat.data); 
  float min_val=100000, max_val=0;
  //(TODO) is there any other ways to copy the data into tensor?
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      res_data[width*y+x] = float(tensor_data(y*width+x)) * 255;
      min_val = std::min(min_val, tensor_data(y*width+x));
      max_val = std::max(max_val, tensor_data(y*width+x));
    }
  }
  std::cout<<"min max tensor value: "<<min_val<<" "<<max_val<<std::endl;
  return res_mat;
}

float get_angle(TextBox& text_box){
  std::vector<cv::Point> points = text_box.get_points();
  float offset_y = points[1].y - points[0].y;
  float offset_x = points[1].x - points[0].x;
  return atan2(offset_y, offset_x);  
}

void get_cropped_extend_image(cv::Mat& image, TextBox& box, cv::Mat& cropped, std::vector<cv::Point>& new_points){
  cv::Point p1, p2;
  box.get_rectangle_box(p1, p2);
  int height = p2.y - p1.y;
  int width = p1.y - p1.x;

  int extend_len = std::max(height, width);
  int minx = std::max(0, p1.x - extend_len);
  int miny = std::max(0, p1.y - extend_len);
  int maxx = std::min(image.cols, p2.x + extend_len);
  int maxy = std::min(image.rows, p2.y + extend_len);
  
  std::vector<cv::Point> points = box.get_points();
  new_points.resize(points.size());
  for(int i=0; i<points.size(); i++){
    new_points[i].x = points[i].x - minx;
    new_points[i].y = points[i].y - miny;
  }
  
  cv::Rect roi(minx, miny, maxx - minx, maxy - miny);
  cropped = image(roi);
}

cv::Point rotate_point(cv::Point& point, float angle, cv::Point& center){
  float new_x = (point.x - center.x) * cos(angle) - (point.y - center.y) * sin(angle) + center.x;
  float new_y = (point.x - center.x) * sin(angle) + (point.y - center.y) * cos(angle) + center.y;
  return cv::Point(new_x, new_y);
}

void rotate_image_and_points(cv::Mat& cropped, std::vector<cv::Point>& points,
      float angle, cv::Mat& rotated_image, std::vector<cv::Point>& rotated_points){
  int height = cropped.rows, width = cropped.cols;
  cv::Point center(width/2, height/2);
  int min_side = std::min(height, width);
  auto M = cv::getRotationMatrix2D(center, angle * 180./PI, 1.0);
  cv::warpAffine(cropped, rotated_image, M, cv::Size(cropped.cols*2, cropped.rows*2));

  //rotate the images
  rotated_points.resize(points.size()); 
  for(int i=0; i<rotated_points.size(); i++){
    rotated_points[i] = rotate_point(points[i], -angle, center);
  }
  //draw_polygon(rotated_image, rotated_points);
  //cv::imwrite("test.jpg", rotated_image);
  
  //crop the word image. It contains some background.
  float extend_ratio_x = 0.05;
  float extend_ratio_y = 0.1;
  int minx = 10000, miny = 10000, maxx = 0, maxy = 0;

  for(auto &point: rotated_points){
    minx = std::min(minx, point.x); 
    miny = std::min(miny, point.y); 
    maxx = std::max(maxx, point.x);
    maxy = std::max(maxy, point.y);
  }
  
  minx = std::max(minx - int(extend_ratio_x * min_side), 0);
  miny = std::max(miny - int(extend_ratio_y * min_side), 0);
  maxx = std::min(maxx + int(extend_ratio_x * min_side), rotated_image.cols);
  maxy = std::min(maxy + int(extend_ratio_y * min_side), rotated_image.rows);

  //crop it
  rotated_image = rotated_image(cv::Rect(minx, miny, maxx-minx, maxy-miny));
  for(auto & ele: rotated_points){
    ele.x -= minx;
    ele.y -= miny;
  }
  
  //draw_polygon(rotated_image, rotated_points);
}

void draw_polygon(cv::Mat& image, std::vector<cv::Point>& points){
  std::vector<cv::Scalar> colors={cv::Scalar(0,0,255), cv::Scalar(0,255,0),
    cv::Scalar(255,0,0), cv::Scalar(255,255,0), cv::Scalar(0,255,255), cv::Scalar(255,0,255)};
  for(int j=0; j<4; j++){
    cv::line(image, points[j], points[(j+1)%4], colors[j%4], 3); 
  }
}

void draw_text_box(cv::Mat& image, TextBox& text_box){
  //draw the polygon
  std::vector<cv::Scalar> colors={cv::Scalar(0,0,255), cv::Scalar(0,255,0),
    cv::Scalar(255,0,0), cv::Scalar(255,255,0), cv::Scalar(0,255,255), cv::Scalar(255,0,255)};
  draw_polygon(image, text_box.get_points());
  //draw text above the left up corner
  cv::Point p1, p2;
  text_box.get_rectangle_box(p1, p2);
  cv::Point draw_loc(std::max(0, p1.x - 10), std::max(0, p1.y - 10));
  cv::putText(image, text_box.get_text(), draw_loc, cv::FONT_HERSHEY_PLAIN, 1.3,  cv::Scalar(0,255,255));
}
