cmake_minimum_required(VERSION 2.8)
project(SceneTextDetection)
set (CMAKE_CXX_STANDARD 11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(SOURCE_FILES main.cpp scene_text_detector.cpp ctc_scene_text_recognizer.cpp utils.cpp scene_text_reader.cpp text_box.cpp)
set(EXECUTABLE DetectText)
#set(EXECUTABLE EndtoEndReading)
#set(EXECUTABLE RecognizeText)

# Add modules
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules")

find_package(TensorFlow REQUIRED)
find_package(Protobuf REQUIRED)
find_package(OpenCV REQUIRED highgui imgproc)

#set(TensorFlow_LIBRARIES "/usr/local/lib/libtensorflow_cc.so" "/usr/local/lib/libtensorflow_framework.so") 
#set(TensorFlow_LIBRARIES "/usr/local/lib/libtensorflow_cc.so" "/usr/local/lib/libtensorflow.so") 
set(TensorFlow_LIBRARIES "/usr/local/lib/libtensorflow.so" "/usr/local/lib/libtensorflow_cc.so" "/usr/local/lib/libtensorflow_framework.so")

message(STATUS "protobuf include dirs:" ${PROTOBUF_INCLUDE_DIRS})
message(STATUS "protobuf libraries:" ${PROTOBUF_LIBRARIES})
message(STATUS "tensorflow include dirs:" ${TensorFlow_INCLUDE_DIRS})
message(STATUS "tensorflow libraries:" ${TensorFlow_LIBRARIES})

# set variables for external dependencies
set(EXTERNAL_DIR "${PROJECT_SOURCE_DIR}/external" CACHE PATH "Location where external dependencies will installed")
set(DOWNLOAD_LOCATION "${EXTERNAL_DIR}/src" CACHE PATH "Location where external projects will be downloaded")

#change the path to ur tesnorflow installed path
#set(NSYNC_DIR "/usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public")

mark_as_advanced(EXTERNAL_DIR DOWNLOAD_LOCATION)

include(Eigen)

set(PROJECT_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIRS} ${EXTERNAL_DIR}/include ${PROTOBUF_INCLUDE_DIRS} ${NSYNC_DIR} ${OpenCV_INCLUDE_DIRS})
set(PROJECT_LIBRARIES ${TensorFlow_LIBRARIES} ${PROTOBUF_LIBRARIES} ${OpenCV_LIBS})
set(PROJECT_DEPENDENCIES Eigen)

include_directories(${PROJECT_INCLUDE_DIRS})
add_library(SceneTextDetector scene_text_detector.cpp utils.cpp)
add_library(CTCSceneTextRecognizer ctc_scene_text_recognizer.cpp utils.cpp)
add_library(SceneTextReader scene_text_reader.cpp scene_text_detector.cpp ctc_scene_text_recognizer.cpp utils.cpp text_box.cpp)
add_executable(${EXECUTABLE} ${SOURCE_FILES})
target_link_libraries(${EXECUTABLE} ${PROJECT_LIBRARIES})
add_dependencies(${EXECUTABLE} ${PROJECT_DEPENDENCIES})
