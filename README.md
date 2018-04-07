# DeepSceneTextReader
This is a c++ project deploy a deep scene text reading. It reads text from natural scene images.

# Prerequsites

The project is written in c++ using tensorflow computational framework. It is tested using tensorflow 1.4. Newer version should be ok too, but not tested.
Please install:

nsync project: https://github.com/google/nsync.git
opencv3.3
protobuf

Please check this project on how to build tensorflow with cmake:
https://github.com/cjweeks/tensorflow-cmake
It greatly helped the process of building this project.


# build process

cd build
cmake ..
make

it will create an excutable DetectText in bin folder.

# Usage:
The excutable could be excuted in three modes:  (1) Detect  (2) Recognize  (3) Detect and Recognize


