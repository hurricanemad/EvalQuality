cmake_minimum_required(VERSION 2.6)
project(evalquality)

find_package(OpenCV REQUIRED)

include_directories(${OPENCV_INCLUDE_DIR})

add_executable(evalquality main.cpp)

target_link_libraries(evalquality ${OpenCV_LIBS})

install(TARGETS evalquality RUNTIME DESTINATION install)
