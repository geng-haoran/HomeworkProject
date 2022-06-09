cmake_minimum_required(VERSION 3.10)

project(Tutorial VERSION 1.0)


add_subdirectory(MathFunctions)

add_executable(Tutorial helloworld.cpp)

target_link_libraries(Tutorial PUBLIC MathFunctions)

target_include_directories(Tutorial PUBLIC "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/MathFunctions")


