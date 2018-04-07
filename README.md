# DeepSceneTextReader
This is a c++ project deploy a deep scene text reading. It reads text from natural scene images.

# Prerequsites

The project is written in c++ using tensorflow computational framework. It is tested using tensorflow 1.4. Newer version should be ok too, but not tested.
Please install:

nsync project: https://github.com/google/nsync.git

opencv3.3

protobuf

eigen

Please check this project on how to build project using tensorflow with cmake:
https://github.com/cjweeks/tensorflow-cmake
It greatly helped the process of building this project.


# build process

cd build
cmake ..
make

it will create an excutable DetectText in bin folder.

# Usage:
The excutable could be excuted in three modes:  (1) Detect  (2) Recognize  (3) Detect and Recognize

## Detect
Download the pretrained detector model and put it in model/
detector_graph=model/Detector_model.pb
image_filename=test_images/test_img1.jpg
output_filename=results/output_image.jpg
mode='detect'

./DetectText --detector_graph=$detector_graph \
   --image_filename=$image_filename --mode=$mode --output_filename=$output_filename

## Recognize
Download the pretrained recognizer model and put it in model/
Download the dictionary file and put it in model

recognizer_graph='model/Recognizer_model.pb'
image_filename=test_images/recognize_image1.jpg
dictionary_filename='model/charset_full.txt'
mode='recognize'

./DetectText --recognizer_graph=$recognizer_graph \
   --image_filename=$image_filename --mode=$mode 

## Detect and Recognize
Download the pretrained detector and recognizer model and put it in model/

detector_graph=model/Detector_model.pb
recognizer_graph='model/Recognizer_model.pb'
dictionary_filename='model/charset_full.txt'
image_filename=test_images/recognize_image1.jpg
output_filename=results/output_image.jpg
mode='recognize'

./DetectText --recognizer_graph=$recognizer_graph --detector_graph=$detector_graph \
   --image_filename=$image_filename --mode=$mode --output_filename=$output_filename 

