
cmake_minimum_required(VERSION 3.10)
project(Tutorial VERSION 1.0)
configure_file(ConfigFile.h.in ConfigFile.h)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_executable(Tutorial helloworld.cpp)
target_include_directories(Tutorial PUBLIC "${PROJECT_BINARY_DIR}")

